#pragma once

#include "enyo_halfka_model.hpp"

namespace Network::EnyoStorage {

alignas(64) extern int16_t input_weights[N_FEATURES * N_HIDDEN];
alignas(64) extern int16_t input_biases[N_HIDDEN];
alignas(64) extern int8_t l1_weights[N_L1 * N_L2];
alignas(64) extern int8_t l1_weights_transposed[N_L1 * N_L2];
alignas(64) extern int8_t l1_weights_sparse[N_L1 * N_L2];
alignas(64) extern int32_t l1_biases[N_L2];
alignas(64) extern float l2_weights[N_L2 * N_L3];
alignas(64) extern float l2_biases[N_L3];
alignas(64) extern float output_weights[MAX_OUTPUT_BUCKETS * MAX_OUTPUT_WIDTH * N_OUTPUT];
alignas(64) extern float output_biases[MAX_OUTPUT_BUCKETS * N_OUTPUT];
alignas(64) extern int16_t quantized_l2_weights[N_L2 * N_L3];
alignas(64) extern int32_t quantized_l2_biases[N_L3];
alignas(64) extern int16_t quantized_output_weights[N_L3 * N_OUTPUT];
alignas(64) extern int32_t quantized_output_bias;

void Clear();
void Activate(const NetworkLayout & layout, DenseLayerFormat format);

} // namespace Network::EnyoStorage
