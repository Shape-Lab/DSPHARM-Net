/****************************************************
 * July 2022
 *
 * Ilwoo Lyu, ilwoolyu@postech.ac.kr
 *
 * 3D Shape Analysis Lab
 * Department of Computer Science and Engineering
 * Pohang University of Science and Technology
 ****************************************************/

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

__global__ void triangle_search_kernel(
	int64_t *id,
	const float *__restrict__ vertex,
	const int64_t *__restrict__ face,
	const int64_t ncand,
	const float *__restrict__ query,
	const float *__restrict__ faceNormal,
	const float *__restrict__ inner,
	const float *__restrict__ area,
	const int64_t nvert,
	const float eps)
{
	const int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nvert)
	{
		id[tid] = -1;
		float cond = eps;
		for (int64_t i = 0; i < ncand; i++)
		{
			const float *n = &faceNormal[i * 3];
			const float *q = &query[tid * 3];
			const float proj = q[0] * n[0] + q[1] * n[1] + q[2] * n[2];

			if (fabs(proj) >= fabs(inner[i]) + eps)
			{
				const float r = inner[i] / proj;
				if (r < 0) continue;
				const float q_proj[3] = {q[0] * r, q[1] * r, q[2] * r};

				const float *a = &vertex[face[i * 3 + 0] * 3];
				const float *b = &vertex[face[i * 3 + 1] * 3];
				const float *c = &vertex[face[i * 3 + 2] * 3];

				const float qa[3] = {a[0] - q_proj[0], a[1] - q_proj[1], a[2] - q_proj[2]};
				const float qb[3] = {b[0] - q_proj[0], b[1] - q_proj[1], b[2] - q_proj[2]};
				const float qc[3] = {c[0] - q_proj[0], c[1] - q_proj[1], c[2] - q_proj[2]};
				const float qbxqc[3] = {qb[1] * qc[2] - qb[2] * qc[1], qb[2] * qc[0] - qb[0] * qc[2], qb[0] * qc[1] - qb[1] * qc[0]};
				const float qcxqa[3] = {qc[1] * qa[2] - qc[2] * qa[1], qc[2] * qa[0] - qc[0] * qa[2], qc[0] * qa[1] - qc[1] * qa[0]};

				float bary[3];
				bary[0] = (qbxqc[0] * n[0] + qbxqc[1] * n[1] + qbxqc[2] * n[2]) / area[i];
				bary[1] = (qcxqa[0] * n[0] + qcxqa[1] * n[1] + qcxqa[2] * n[2]) / area[i];
				bary[2] = 1 - bary[0] - bary[1];

				if (bary[0] >= cond && bary[1] >= cond && bary[2] >= cond)
				{
					id[tid] = i;
					cond = (bary[0] < bary[1]) ? bary[0] : bary[1];
					if (bary[2] < cond)
						cond = bary[2];
					if (cond >= 0)
						break;
				}
			}
		}
	}
}

at::Tensor triangle_search_cuda(
	const at::Tensor &vertex,
	const at::Tensor &face,
	const at::Tensor &query,
	const at::Tensor &faceNormal,
	const at::Tensor &inner,
	const at::Tensor &area,
	const float eps)
{
	at::TensorArg vertex_t{vertex, "vertex", 1}, face_t{face, "face", 2},
		query_t{query, "query", 3}, faceNormal_t{faceNormal, "faceNormal", 4},
		inner_t{inner, "inner", 5}, area_t{area, "area", 6};
	at::CheckedFrom c = "triangle_search_cuda";
	at::checkAllSameGPU(c, {vertex_t, face_t, query_t, faceNormal_t, inner_t, area_t});

	at::cuda::CUDAGuard device_guard(vertex.device());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	TORCH_CHECK(face.size(0) == faceNormal.size(0));
	TORCH_CHECK(face.size(0) == inner.size(0));
	TORCH_CHECK(face.size(0) == area.size(0));
	TORCH_CHECK(eps >= 0);

	const int64_t nquery = query.size(0);

	const size_t threads = 256;
	const size_t blocks = (nquery + threads - 1) / threads;

	const int device_id = vertex.device().index();
	auto opt_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_id);
	auto fid = at::empty({nquery}, opt_long);

	triangle_search_kernel<<<blocks, threads, 0, stream>>>(
		fid.data_ptr<int64_t>(),
		vertex.data_ptr<float>(),
		face.data_ptr<int64_t>(),
		face.size(0),
		query.data_ptr<float>(),
		faceNormal.data_ptr<float>(),
		inner.data_ptr<float>(),
		area.data_ptr<float>(),
		nquery,
		-eps);

	AT_CUDA_CHECK(cudaGetLastError());

	return fid;
}
