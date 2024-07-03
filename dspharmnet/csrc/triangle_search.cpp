/****************************************************
 * July 2022
 *
 * Ilwoo Lyu, ilwoolyu@postech.ac.kr
 *
 * 3D Shape Analysis Lab
 * Department of Computer Science and Engineering
 * Pohang University of Science and Technology
 ****************************************************/

#include <torch/extension.h>

at::Tensor triangle_search_cuda(
	const at::Tensor &vertex,
	const at::Tensor &face,
	const at::Tensor &query,
	const at::Tensor &faceNormal,
	const at::Tensor &inner,
	const at::Tensor &area,
	const float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("query", &triangle_search_cuda);
}
