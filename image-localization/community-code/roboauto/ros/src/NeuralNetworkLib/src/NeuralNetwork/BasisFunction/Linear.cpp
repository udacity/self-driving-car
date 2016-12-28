/*
 *  Copyright 2016 Tomas Cernik, Tom.Cernik@gmail.com
 *  All rights reserved.
 *
 *  This file is part of NeuralNetworkLib
 *
 *  NeuralNetworkLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  NeuralNetworkLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NeuralNetworkLib.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <NeuralNetwork/BasisFunction/Linear.h>

float NeuralNetwork::BasisFunction::Linear::operator()(const std::vector<float> &weights, const std::vector<float> &input) const {
	assert(input.size() >= weights.size());
	std::size_t weightsSize=weights.size();

#ifdef USE_AVX

	std::size_t alignedPrev=weightsSize-weightsSize%8;

	const float* weightsData=weights.data();
	const float* inputData=input.data();

	union {
		__m256 avx;
		float f[8];
	} partialSolution;

	partialSolution.avx=_mm256_setzero_ps();

#ifndef USE_FMA
	__m256 tmp;
#endif

	for(size_t k=0;k<alignedPrev;k+=8) {
		//TODO: assignement!! -- possible speedup
#ifdef USE_FMA
		partialSolution.avx=_mm256_fmadd_ps(_mm256_loadu_ps(weightsData+k),_mm256_loadu_ps(inputData+k),partialSolution.avx);
#else
		tmp=_mm256_mul_ps(_mm256_loadu_ps(weightsData+k),_mm256_loadu_ps(inputData+k));
		partialSolution.avx=_mm256_add_ps(tmp,partialSolution.avx);
#endif
	}

	for(size_t k=alignedPrev;k<weightsSize;k++) {
#ifdef USE_FMA
		partialSolution.avx=_mm256_fmadd_ps(_mm256_set_ps(weightsData[k],0,0,0,0,0,0,0),_mm256_set_ps(inputData[k],0,0,0,0,0,0,0),partialSolution.avx);
#else
		tmp=_mm256_mul_ps(_mm256_set_ps(weightsData[k],0,0,0,0,0,0,0),_mm256_set_ps(inputData[k],0,0,0,0,0,0,0));
		partialSolution.avx=_mm256_add_ps(tmp,partialSolution.avx);
#endif
	}

	partialSolution.avx = _mm256_add_ps(partialSolution.avx,  _mm256_permute2f128_ps(partialSolution.avx , partialSolution.avx , 1));
	partialSolution.avx = _mm256_hadd_ps(partialSolution.avx, partialSolution.avx);
	partialSolution.avx = _mm256_hadd_ps(partialSolution.avx, partialSolution.avx);

	return partialSolution.f[0];

#elif USE_SSE

	std::size_t alignedPrev=weightSize-weightSize%4;

	const float* weightsData=weights.data();
	const float* inputData=input.data();
	vec4f partialSolution;
	partialSolution.sse =_mm_setzero_ps();

	//TODO prefetch ??
	for(register size_t k=0;k<alignedPrev;k+=4) {
	partialSolution.sse=_mm_add_ps(partialSolution.sse,_mm_mul_ps(_mm_load_ps(weightsData+k),_mm_load_ps(inputData+k)));
	}

	for(register size_t k=alignedPrev;k<weightSize;k++) {
	partialSolution.sse=_mm_add_ps(partialSolution.sse,_mm_mul_ps(_mm_load_ss(weightsData+k),_mm_load_ss(inputData+k)));
	}

	#ifdef USE_SSE2 //pre-SSE3 solution
		partialSolution.sse= _mm_add_ps(_mm_movehl_ps(partialSolution.sse, partialSolution.sse), partialSolution.sse);
		partialSolution.sse=_mm_add_ss(partialSolution.sse, _mm_shuffle_ps(partialSolution.sse,partialSolution.sse, 1));
	#else
		partialSolution.sse = _mm_hadd_ps(partialSolution.sse, partialSolution.sse);
		partialSolution.sse = _mm_hadd_ps(partialSolution.sse, partialSolution.sse);
	#endif
	return partialSolution.f[0];
#else

	register float tmp = 0;
	for(size_t k=0;k<weightSize;k++) {
		tmp+=input[k]*weights[k];
	}
	return tmp;
#endif
}
