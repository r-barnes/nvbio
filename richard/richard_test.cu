#include <nvbio-test/alignment_test_utils.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/cuda/ldg.h>
#include <nvbio/basic/cached_iterator.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/vector_view.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/basic/dna.h>
#include <nvbio/alignment/alignment.h>
#include <nvbio/alignment/batched.h>
#include <nvbio/alignment/sink.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

using namespace nvbio;

using namespace nvbio::aln;

enum { CACHE_SIZE = 32 };
typedef nvbio::lmem_cache_tag<CACHE_SIZE>                                       lmem_cache_tag_type;
typedef nvbio::uncached_tag                                                     uncached_tag_type;

//
// An alignment stream class to be used in conjunction with the BatchAlignmentScore class
//
template <typename t_aligner_type, uint32 M, uint32 N, typename cache_type = lmem_cache_tag_type>
struct AlignmentStream
{
    typedef t_aligner_type                                                          aligner_type;

    typedef nvbio::cuda::ldg_pointer<uint32>                                        storage_iterator;

    typedef nvbio::PackedStringLoader<storage_iterator,4,false,cache_type>          pattern_loader_type;
    typedef typename pattern_loader_type::input_iterator                            uncached_pattern_iterator;
    typedef typename pattern_loader_type::iterator                                  pattern_iterator;
    typedef nvbio::vector_view<pattern_iterator>                                    pattern_string;

    typedef nvbio::PackedStringLoader<storage_iterator,2,false,cache_type>          text_loader_type;
    typedef typename text_loader_type::input_iterator                               uncached_text_iterator;
    typedef typename text_loader_type::iterator                                     text_iterator;
    typedef nvbio::vector_view<text_iterator>                                       text_string;

    // an alignment context
    struct context_type
    {
        int32                   min_score;
        aln::BestSink<int32>    sink;
    };
    // a container for the strings to be aligned
    struct strings_type
    {
        pattern_loader_type     pattern_loader;
        text_loader_type        text_loader;
        pattern_string          pattern;
        trivial_quality_string  quals;
        text_string             text;
    };

    // constructor
    AlignmentStream(
        aligner_type        _aligner,
        const uint32        _count,
        const uint32*       _patterns,
        const uint32*       _text,
               int16*       _scores) :
        m_aligner( _aligner ), m_count(_count), m_patterns(storage_iterator(_patterns)), m_text(storage_iterator(_text)), m_scores(_scores) {}

    // get the aligner
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const aligner_type& aligner() const { return m_aligner; };

    // return the maximum pattern length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 max_pattern_length() const { return M; }

    // return the maximum text length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 max_text_length() const { return N; }

    // return the stream size
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_count; }

    // return the i-th pattern's length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pattern_length(const uint32 i, context_type* context) const { return M; }

    // return the i-th text's length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 text_length(const uint32 i, context_type* context) const { return N; }

    // initialize the i-th context
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool init_context(
        const uint32    i,
        context_type*   context) const
    {
        context->min_score = Field_traits<int32>::min();
        return true;
    }

    // initialize the i-th context
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void load_strings(
        const uint32        i,
        const uint32        window_begin,
        const uint32        window_end,
        const context_type* context,
              strings_type* strings) const
    {
        strings->pattern = pattern_string( M,
            strings->pattern_loader.load(
                m_patterns + i * M,
                M,
                make_uint2( window_begin, window_end ),
                false ) );

        strings->text = text_string( N, strings->text_loader.load( m_text + i * N, N ) );
    }

    // handle the output
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void output(
        const uint32        i,
        const context_type* context) const
    {
        // copy the output score
        m_scores[i] = context->sink.score;
    }

    aligner_type                m_aligner;
    uint32                      m_count;
    uncached_pattern_iterator   m_patterns;
    uncached_text_iterator      m_text;
    int16*                      m_scores;
};

//
// A class for making a single alignment test, testing both scoring and traceback
//
struct SingleTest
{
    thrust::host_vector<uint8>   str_hvec;
    thrust::host_vector<uint8>   ref_hvec;
    thrust::device_vector<uint8> str_dvec;
    thrust::device_vector<uint8> ref_dvec;
    thrust::device_vector<float> temp_dvec;
    thrust::device_vector<float> score_dvec;
    thrust::device_vector<uint2> sink_dvec;

    // test banded alignment
    //
    // \param test              test name
    // \param aligner           alignment algorithm
    // \param ref_alignment     reference alignment string
    //
    template <uint32 BLOCKDIM, uint32 BAND_LEN, const uint32 N, const uint32 M, typename aligner_type>
    void banded(const char* test, const aligner_type aligner, const char* ref_alignment)
    {
        NVBIO_VAR_UNUSED const uint32 CHECKPOINTS = 32u;

        const uint8* str_hptr = nvbio::raw_pointer( str_hvec );
        const uint8* ref_hptr = nvbio::raw_pointer( ref_hvec );

        const int32 ref_score = ref_banded_sw<M,N,BAND_LEN>( str_hptr, ref_hptr, 0u, aligner );

        aln::BestSink<int32> sink;
        aln::banded_alignment_score<BAND_LEN>(
            aligner,
            vector_view<const uint8*>( M, str_hptr ),
            trivial_quality_string(),
            vector_view<const uint8*>( N, ref_hptr ),
            -1000,
            sink );

        const int32 cpu_score = sink.score;
        if (cpu_score != ref_score)
        {
            log_error(stderr, "    expected %s score %d, got: %d\n", test, ref_score, cpu_score);
            exit(1);
        }

        TestBacktracker backtracker;
        backtracker.clear();

        const Alignment<int32> aln = aln::banded_alignment_traceback<BAND_LEN,1024u,CHECKPOINTS>(
            aligner,
            vector_view<const uint8*>( M, str_hptr ),
            trivial_quality_string(),
            vector_view<const uint8*>( N, ref_hptr ),
            -1000,
            backtracker );

        const int32 aln_score = backtracker.score( aligner, aln.source.x, str_hptr, ref_hptr );
        const std::string aln_string = rle( backtracker.aln ).c_str();
        if (aln_score != ref_score)
        {
            log_error(stderr, "    expected %s backtracking score %d, got %d\n", ref_score, aln_score);
            log_error(stderr, "    %s - %d - [%u, %u] x [%u, %u]\n", aln_string.c_str(), aln.score, aln.source.x, aln.sink.x, aln.source.y, aln.sink.y);
            exit(1);
        }
        fprintf(stderr, "    %15s : ", test);
        fprintf(stderr, "%d - %s - [%u:%u] x [%u:%u]\n", aln.score, aln_string.c_str(), aln.source.x, aln.sink.x, aln.source.y, aln.sink.y);
        if (strcmp( ref_alignment, aln_string.c_str() ) != 0)
        {
            log_error(stderr, "    expected %s, got %s\n", ref_alignment, aln_string.c_str());
            exit(1);
        }
    }
};

// execute and time the batch_banded_score<scheduler> algorithm for all possible schedulers
//
template <uint32 BAND_LEN, uint32 N, uint32 M, typename aligner_type>
void batch_banded_score_profile_all(
    const aligner_type              aligner,
    const uint32                    n_tasks,
    thrust::device_vector<uint32>&  pattern_dvec,
    thrust::device_vector<uint32>&  text_dvec,
    thrust::device_vector<int16>&   score_dvec)
{
    // create a stream
    typedef AlignmentStream<aligner_type,M,N> stream_type;
    stream_type stream(
        aligner,
        n_tasks,
        nvbio::raw_pointer( pattern_dvec ),
        nvbio::raw_pointer( text_dvec ),
        nvbio::raw_pointer( score_dvec ) );

    // setup a batch
    //TODO: Can also use: DeviceStagedThreadScheduler. Maybe also DeviceWarpScheduler?
    typedef aln::BatchedBandedAlignmentScore<BAND_LEN,stream_type, DeviceThreadScheduler> batch_type;
    batch_type batch;

    // alloc all the needed temporary storage
    const uint64 temp_size = batch_type::max_temp_storage(
        stream.max_pattern_length(),
        stream.max_text_length(),
        stream.size() );

    thrust::device_vector<uint8> temp_dvec( temp_size );

    Timer timer;
    timer.start();
    batch.enact( stream, temp_size, nvbio::raw_pointer( temp_dvec ) );
    cudaDeviceSynchronize();
    timer.stop();

    const float time = timer.seconds();

    fprintf(stderr,"  %5.1f", 1.0e-9f * float(n_tasks*uint64(BAND_LEN*M))*(1/time) );
    fprintf(stderr, " GCUPS\n");
}



int main(int argc, char* argv[])
{
    NVBIO_VAR_UNUSED uint32 N_WARP_TASKS     = 4096;
                     uint32 N_THREAD_TASKS   = 128*1024;

    fprintf(stderr,"testing alignment... started\n");

    const uint32 BAND_LEN = 15u;
    const uint32 N_TASKS  = N_THREAD_TASKS;
    const uint32 M = 150;
    const uint32 N = M+BAND_LEN;

    const uint32 M_WORDS = (M + 7)  >> 3;
    const uint32 N_WORDS = (N + 15) >> 4;

    thrust::host_vector<uint32> str( M_WORDS * N_TASKS );
    thrust::host_vector<uint32> ref( N_WORDS * N_TASKS );

    LCG_random rand;
    fill_packed_stream<4u>( rand, 4u, M * N_TASKS, nvbio::raw_pointer( str ) );
    fill_packed_stream<2u>( rand, 4u, N * N_TASKS, nvbio::raw_pointer( ref ) );

    thrust::device_vector<uint32> str_dvec( str );
    thrust::device_vector<uint32> ref_dvec( ref );
    thrust::device_vector<int16>  score_dvec( N_TASKS );

    fprintf(stderr,"  testing banded Smith-Waterman scoring speed...\n");
    //Also aln::SEMI_GLOBAL, aln::GLOBAL
    fprintf(stderr,"    %15s : ", "local");
    batch_banded_score_profile_all<BAND_LEN,N,M>(
        make_smith_waterman_aligner<aln::LOCAL>( aln::SimpleSmithWatermanScheme(2,-1,-1,-1) ),
        N_TASKS,
        str_dvec,
        ref_dvec,
        score_dvec
    );

    return 0;
}
