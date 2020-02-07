void banded_traceback_best(
    const uint32                                                aln_idx,
    const uint32                                                count,
    const uint32*                                               idx,
          io::Alignment*                                        best_data,
    const uint32                                                best_stride,
    const uint32                                                band_len,
    const TracebackPipelineState<EditDistanceScoringScheme>&    pipeline,
    const ParamsPOD&                                            params)
{
    if (aln_idx)
        banded_traceback_best_t<1>( count, idx, best_data, best_stride, band_len, pipeline, params );
    else
        banded_traceback_best_t<0>( count, idx, best_data, best_stride, band_len, pipeline, params );
}



//
// finish a batch of alignment calculations
//
void finish_alignment_best(
    const uint32                                                    aln_idx,
    const uint32                                                    count,
    const uint32*                                                   idx,
          io::Alignment*                                            best_data,
    const uint32                                                    best_stride,
    const uint32                                                    band_len,
    const TracebackPipelineState<SmithWatermanScoringScheme<> >&    pipeline,
    const SmithWatermanScoringScheme<>                              scoring_scheme,
    const ParamsPOD&                                                params)
{
    if (aln_idx)
        finish_alignment_best_t<1>( count, idx, best_data, best_stride, band_len, pipeline, scoring_scheme, params );
    else
        finish_alignment_best_t<0>( count, idx, best_data, best_stride, band_len, pipeline, scoring_scheme, params );
}









template <typename scoring_tag>
void Aligner::best_approx(
    const Params&                           params,
    const fmi_type                          fmi,
    const rfmi_type                         rfmi,
    const UberScoringScheme&                input_scoring_scheme,
    const io::SequenceDataDevice&           reference_data,
    const io::FMIndexDataDevice&            driver_data,
    const io::SequenceDataDevice&           read_data,
    io::HostOutputBatchSE&                  cpu_batch,
    Stats&                                  stats)
{
    // cast the genome to use proper iterators
    const genome_view_type          genome_view( plain_view( reference_data ) );
    const genome_access_type        genome_access( genome_view );
    const uint32                    genome_len = genome_access.bps();
    const genome_iterator           genome     = genome_access.sequence_stream();

    // prepare the scoring system
    typedef typename ScoringSchemeSelector<scoring_tag>::type           scoring_scheme_type;
    typedef typename scoring_scheme_type::threshold_score_type          threshold_score_type;

    scoring_scheme_type scoring_scheme = ScoringSchemeSelector<scoring_tag>::scheme( input_scoring_scheme );

    threshold_score_type threshold_score = scoring_scheme.threshold_score();
    //const int32          score_limit     = scoring_scheme_type::worst_score;

    // start timing
    Timer timer;
    Timer global_timer;
    nvbio::cuda::Timer device_timer;

    const uint32 count = read_data.size();
    const uint32 band_len = band_length( params.max_dist );

    // create a device-side read batch
    const read_view_type  reads_view = plain_view( read_data );
    const read_batch_type reads( reads_view );

    // initialize best-alignments
    init_alignments( reads, threshold_score, best_data_dptr, BATCH_SIZE );

    seed_queues.resize( count );

    thrust::copy(
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(0u) + count,
        seed_queues.in_queue.begin() );

    //
    // Similarly to Bowtie2, we perform a number of seed & extension passes.
    // Whether a read is re-seeded is determined at run time based on seed hit and
    // alignment statistics.
    // Hence, the number of reads actively processed in each pass can vary substantially.
    // In order to keep the cores and all its lanes busy, we use a pair of input & output
    // queues to compact the set of active reads in each round, swapping them at each
    // iteration.
    //

    // filter away reads that map exactly
    if (0)
    {
        // initialize output seeding queue size
        seed_queues.clear_output();

        //
        // perform whole read mapping
        //
        {
            timer.start();
            device_timer.start();

            // initialize the seed hit counts
            hit_deques.clear_deques();

            SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

            log_debug(stderr, "[%u]     map\n", ID);
            map_whole_read(
                reads, fmi, rfmi,
                seed_queues.device_view(),
                reseed_dptr,
                hits,
                params,
                params.fw,
                params.rc );

            optional_device_synchronize();
            nvbio::cuda::check_error("mapping kernel");

            device_timer.stop();
            timer.stop();
            stats.map.add( seed_queues.in_size, timer.seconds(), device_timer.seconds() );
        }

        best_approx_score<scoring_tag>(
            params,
            fmi,
            rfmi,
            scoring_scheme,
            reference_data,
            driver_data,
            read_data,
            uint32(-1),
            seed_queues.in_size,
            seed_queues.raw_input_queue(),
            stats );

        log_verbose( stderr, "[%u]     %.1f %% reads map exactly\n", ID, 100.0f * float(count - seed_queues.output_size())/float(count) );

        // copy the reads that need reseeding
        seed_queues.out_size[0] = nvbio::copy_flagged(
            seed_queues.in_size,
            (uint32*)seed_queues.raw_input_queue(),
            reseed_dptr,
            (uint32*)seed_queues.raw_output_queue(),
            temp_dvec );

        // swap input & output queues
        seed_queues.swap();
    }

    for (uint32 seeding_pass = 0; seeding_pass < params.max_reseed+1; ++seeding_pass)
    {
        // check whether the input queue is empty
        if (seed_queues.in_size == 0)
            break;

        // initialize output seeding queue size
        seed_queues.clear_output();

        //
        // perform mapping
        //
        {
            timer.start();
            device_timer.start();

            hit_deques.clear_deques();

            SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

            log_debug(stderr, "[%u]     map\n", ID);
            map(
                reads, fmi, rfmi,
                seeding_pass, seed_queues.device_view(),
                reseed_dptr,
                hits,
                params,
                params.fw,
                params.rc );

            optional_device_synchronize();
            nvbio::cuda::check_error("mapping kernel");

            device_timer.stop();
            timer.stop();
            stats.map.add( seed_queues.in_size, timer.seconds(), device_timer.seconds() );

            // check if we need to persist this seeding pass
            if (params.persist_batch   == -1 || batch_number == (uint32) params.persist_batch &&
                params.persist_seeding == -1 || seeding_pass == (uint32) params.persist_seeding)
                persist_hits( params.persist_file, "hits", 0u, count, hit_deques );
        }

        // take some stats on the hits we got
        if (seeding_pass == 0 && params.keep_stats)
            keep_stats( reads.size(), stats );

        best_approx_score<scoring_tag>(
            params,
            fmi,
            rfmi,
            scoring_scheme,
            reference_data,
            driver_data,
            read_data,
            seeding_pass,
            seed_queues.in_size,
            seed_queues.raw_input_queue(),
            stats );

        // mark unaligned reads
        mark_unaligned(
            seed_queues.in_size,
            seed_queues.raw_input_queue(),
            best_data_dptr,
            reseed_dptr );

        // copy the reads that need reseeding
        seed_queues.out_size[0] = nvbio::copy_flagged(
            seed_queues.in_size,
            (uint32*)seed_queues.raw_input_queue(),
            reseed_dptr,
            (uint32*)seed_queues.raw_output_queue(),
            temp_dvec );

        // swap input & output queues
        seed_queues.swap();
    }

    //
    // At this point, for each read we have the scores and rough alignment positions of the
    // best two alignments: to compute the final results we need to backtrack the DP extension,
    // and compute accessory CIGARS and MD strings.
    //

    // compute mapq
    {
        log_debug(stderr, "[%u]     compute mapq\n", ID);
        typedef BowtieMapq2< SmithWatermanScoringScheme<> > mapq_evaluator_type;

        mapq_evaluator_type mapq_eval( input_scoring_scheme.sw );

        MapqFunctorSE<mapq_evaluator_type,read_view_type> mapq_functor( mapq_eval, best_data_dptr, BATCH_SIZE, reads_view );

        nvbio::transform<device_tag>(
            count,
            thrust::make_counting_iterator<uint32>(0),
            mapq_dvec.begin(),
            mapq_functor );
    }

    TracebackPipelineState<scoring_scheme_type> traceback_state(
        reads,
        reads,
        genome_len,
        genome,
        scoring_scheme,
        *this );

    //
    // perform backtracking and compute cigars for the best alignments
    //
    {
        // initialize cigars & MDS pools
        cigar.clear();
        mds.clear();

        timer.start();
        device_timer.start();

        log_debug(stderr, "[%u]     backtrack\n", ID);
        banded_traceback_best(
            0u,
            count,
            NULL,
            best_data_dptr,
            BATCH_SIZE,
            band_len,
            traceback_state,
            params );

        optional_device_synchronize();
        nvbio::cuda::check_error("backtracking kernel");
        if (cigar.has_overflown())
            throw nvbio::runtime_error("CIGAR vector overflow\n");

        device_timer.stop();
        timer.stop();
        stats.backtrack.add( count, timer.seconds(), device_timer.seconds() );

        timer.start();
        device_timer.start();

        log_debug(stderr, "[%u]     alignment\n", ID);
        finish_alignment_best(
            0u,
            count,
            NULL,
            best_data_dptr,
            BATCH_SIZE,
            band_len,
            traceback_state,
            input_scoring_scheme.sw,    // always use Smith-Waterman for the final scoring of the found alignments
            params );

        optional_device_synchronize();
        nvbio::cuda::check_error("alignment kernel");
        if (mds.has_overflown())
            throw nvbio::runtime_error("MDS vector overflow\n");

        device_timer.stop();
        timer.stop();
        stats.finalize.add( count, timer.seconds(), device_timer.seconds() );
    }

    // wrap the results in a DeviceOutputBatchSE and process
    if (output_file)
    {
        timer.start();

        io::DeviceOutputBatchSE gpu_batch(
            count,
            best_data_dvec,
            io::DeviceCigarArray(cigar, cigar_coords_dvec),
            mds,
            mapq_dvec);

        cpu_batch.readback( gpu_batch );

        timer.stop();
        stats.alignments_DtoH.add( count, timer.seconds() );

        timer.start();

        log_debug(stderr, "[%u]     output\n", ID);
        output_file->process( cpu_batch );

        timer.stop();
        stats.io.add( count, timer.seconds() );
    }

    // keep alignment stats
    {
        log_debug(stderr, "[%u]    track stats\n", ID);
        thrust::host_vector<io::Alignment> h_best_data( BATCH_SIZE*2 );
        thrust::host_vector<uint8>         h_mapq( BATCH_SIZE );

        thrust::copy( best_data_dvec.begin(),              best_data_dvec.begin() + count,              h_best_data.begin() );
        thrust::copy( best_data_dvec.begin() + BATCH_SIZE, best_data_dvec.begin() + count + BATCH_SIZE, h_best_data.begin() + BATCH_SIZE );
        thrust::copy( mapq_dvec.begin(),                   mapq_dvec.begin()      + count,              h_mapq.begin() );

        for (uint32 i = 0; i < count; ++i)
        {
            const io::BestAlignments best( h_best_data[i], h_best_data[i + BATCH_SIZE] );
            const uint8 mapq = h_mapq[i];

            stats.track_alignment_statistics( &stats.mate1, best, mapq );
        }
    }
}


void best_approx_sw(
          Aligner&                  aligner,
    const Params&                   params,
    const FMIndexDef::type          fmi,
    const FMIndexDef::type          rfmi,
    const UberScoringScheme&        scoring_scheme,
    const io::SequenceDataDevice&   reference_data,
    const io::FMIndexDataDevice&    driver_data,
    io::SequenceDataDevice&         read_data,
    io::HostOutputBatchSE&          cpu_batch,
    Stats&                          stats)
{
    aligner.best_approx<smith_waterman_scoring_tag>(
        params,
        fmi,
        rfmi,
        scoring_scheme,
        reference_data,
        driver_data,
        read_data,
        cpu_batch,
        stats );
}