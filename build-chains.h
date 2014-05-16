/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#include <nvbio/io/reads/reads.h>

/// build chains for the current pipeline::chunk of reads
///
void build_chains(struct pipeline_state *pipeline, const nvbio::io::ReadDataDevice *batch);

/// filter chains for the current pipeline::chunk of reads
///
void filter_chains(struct pipeline_state *pipeline, const nvbio::io::ReadDataDevice *batch);
