#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <vector>
#include <numeric>
#include <unordered_map>

#if __STDCPP_DEFAULT_NEW_ALIGNMENT__ < 16
#error "Alignment of heap-allocated pointers should be at least 16"
#endif

using Mat8x8Byte = const uint8_t[8][8];
using Mat8x8Float = const float[8][8];
// Assume matrix data is sequential
using MatGenByte = const uint8_t*;
using MatGenFloat = const float*;

uint32_t mat_dot_prod_64(Mat8x8Byte a, Mat8x8Byte b);
uint32_t mat_dot_prod_gen(MatGenByte a, MatGenByte b, int count);
float matf_dot_prod_64(Mat8x8Float a, Mat8x8Float b);
float matf_dot_prod_gen(MatGenFloat a, MatGenFloat b, int count);

extern "C"
{
	uint32_t mat_dot_prod_64_masm_nommx(Mat8x8Byte a, Mat8x8Byte b);
	uint32_t mat_dot_prod_64_masm_mmx(Mat8x8Byte a, Mat8x8Byte b);
	uint32_t mat_dot_prod_64_masm_sse_32bitsum(Mat8x8Byte a, Mat8x8Byte b);
	uint32_t mat_dot_prod_64_masm_sse_16bitsum(Mat8x8Byte a, Mat8x8Byte b);
	uint32_t mat_dot_prod_gen_masm_nommx(MatGenByte a, MatGenByte b, int count);
	uint32_t mat_dot_prod_gen_masm_mmx(MatGenByte a, MatGenByte b, int count);
	uint32_t mat_dot_prod_gen_masm_sse_32bitsum(MatGenByte a, MatGenByte b, int count);
	uint32_t mat_dot_prod_gen_masm_sse_16bitsum(MatGenByte a, MatGenByte b, int count);

	float matf_dot_prod_64_masm_sse(Mat8x8Float a, Mat8x8Float b);
	float matf_dot_prod_64_masm_sse_mulps(Mat8x8Float a, Mat8x8Float b);
	float matf_dot_prod_gen_masm_sse(MatGenFloat a, MatGenFloat b, int count);
	float matf_dot_prod_gen_masm_sse_mulps(MatGenFloat a, MatGenFloat b, int count);
}

constexpr int BENCH_REPEATS = 1000000;

uint32_t mat_dot_prod_64(Mat8x8Byte a, Mat8x8Byte b)
{
	uint32_t sum = 0;
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j)
			sum += static_cast<uint32_t>(a[i][j])* b[i][j];
	return sum;
}

uint32_t mat_dot_prod_gen(MatGenByte a, MatGenByte b, int count)
{
	uint32_t sum = 0;
	for (int i = 0; i < count; ++i)
		sum += static_cast<uint32_t>(a[i])* b[i];
	return sum;
}

float matf_dot_prod_64(Mat8x8Float a, Mat8x8Float b)
{
	// In non-float variants we didn't use
	// std::inner_product because of type conversion
	return std::inner_product((float*)a, (float*)a + 64, (float*)b, 0.f);
}

float matf_dot_prod_gen(MatGenFloat a, MatGenFloat b, int count)
{
	return std::inner_product(a, a + count, b, 0.f);
}

template<typename Fn, typename... Params>
void bench(std::string_view description,
	Fn mulFn, Params&&... params)
{
	using namespace std::chrono;
	using ResultType = std::invoke_result_t<Fn, Params...>;
	constexpr bool ResultVoid = std::is_same_v<ResultType, void>;
	using ResultOutType = typename std::conditional_t<ResultVoid,
		std::string_view, volatile ResultType>;

	ResultOutType result;
	if constexpr (ResultVoid)
		result = "None";

	auto start = high_resolution_clock::now();
	for (int i = 0; i < BENCH_REPEATS; ++i)
		if constexpr (ResultVoid)
			mulFn(std::forward<Params>(params)...);
		else
			result = mulFn(std::forward<Params>(params)...);
	auto end = high_resolution_clock::now();

	std::cout << description << std::endl;
	std::cout << "    Last result: " << result << std::endl;
	std::cout << "    Time: " << duration_cast<microseconds>(
		end - start).count() << " microseconds" << std::endl;
}

int main()
{
	// int y = 0;
	// bench("test", [&y] {++y; });
	// std::cout << y << std::endl;

	alignas(16) constexpr Mat8x8Byte a = {
		{0, 0, 0, 0, 3, 6, 0, 0},
		{3, 0, 1, 4, 0, 1, 0, 3},
		{0, 8, 2, 0, 0, 0, 0, 0},
		{6, 0, 6, 0, 0, 0, 0, 7},
		{0, 2, 0, 0, 0, 0, 0, 6},
		{0, 6, 4, 0, 0, 5, 0, 4},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 2}
	};
	alignas(16) constexpr Mat8x8Byte b = {
		{0, 3, 0, 0, 3, 4, 0, 0},
		{0, 0, 2, 0, 0, 0, 0, 0},
		{0, 1, 8, 0, 0, 8, 0, 5},
		{2, 0, 9, 0, 5, 0, 0, 0},
		{0, 4, 3, 0, 8, 0, 0, 7},
		{0, 1, 5, 5, 0, 0, 0, 8},
		{7, 0, 3, 0, 3, 0, 9, 4},
		{0, 0, 2, 4, 0, 0, 0, 0}
	};
	alignas(16) constexpr Mat8x8Float af = {
		{0, 0, 0, 0, 3, 6, 0, 0},
		{3, 0, 1, 4, 0, 1, 0, 3},
		{0, 8, 2, 0, 0, 0, 0, 0},
		{6, 0, 6, 0, 0, 0, 0, 7},
		{0, 2, 0, 0, 0, 0, 0, 6},
		{0, 6, 4, 0, 0, 5, 0, 4},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 2}
	};
	alignas(16) constexpr Mat8x8Float bf = {
		{0, 3, 0, 0, 3, 4, 0, 0},
		{0, 0, 2, 0, 0, 0, 0, 0},
		{0, 1, 8, 0, 0, 8, 0, 5},
		{2, 0, 9, 0, 5, 0, 0, 0},
		{0, 4, 3, 0, 8, 0, 0, 7},
		{0, 1, 5, 5, 0, 0, 0, 8},
		{7, 0, 3, 0, 3, 0, 9, 4},
		{0, 0, 2, 4, 0, 0, 0, 0}
	};

	std::cout << "All calculations are done "
		<< BENCH_REPEATS << " times" << std::endl;
	std::cout << std::endl;

	std::cout << "Dot-multiplying two 8x8 matrices having 64 byte elements" << std::endl;
	bench("C++ dot product 64 byte values: ",
		mat_dot_prod_64, a, b);
	bench("ASM dot product 64 byte values no extensions: ",
		mat_dot_prod_64_masm_nommx, a, b);
	bench("ASM dot product 64 byte values mmx: ",
		mat_dot_prod_64_masm_mmx, a, b);
	bench("ASM dot product 64 byte values sse 32-bit sum: ",
		mat_dot_prod_64_masm_sse_32bitsum, a, b);
	bench("ASM dot product 64 byte values sse 16-bit sum: ",
		mat_dot_prod_64_masm_sse_16bitsum, a, b);
	std::cout << std::endl;

	std::cout << "Dot-multiplying two 8x8 matrices having 64 float elements" << std::endl;
	bench("C++ dot product 64 float values: ",
		matf_dot_prod_64, af, bf);
	bench("C++ dot product 64 float values sse dpps: ",
		matf_dot_prod_64_masm_sse, af, bf);
	bench("C++ dot product 64 float values sse mulps: ",
		matf_dot_prod_64_masm_sse_mulps, af, bf);
	std::cout << std::endl;

	std::string again = "n";
	do
	{
		std::string test;
		std::cout << "Do you want to test\n"
		             "    dot product for bytes (1),\n"
		             "    dot product for floats (2),\n"
		             "    coef product for floats (otherwise): ";
		std::getline(std::cin, test);

		std::cout << "Enter filename for general matrices: ";
		std::string filename;
		do
		{
			if (!filename.empty())
				std::cout << "File does not exist, enter again please: ";
			std::getline(std::cin, filename);
		} while (!std::filesystem::exists(filename));

		std::ifstream ifs(filename);
		int rows, columns;
		ifs >> rows >> columns;
		const int count = rows * columns;

		if (test == "1")
		{
			std::cout << "Dot-multiplying two " << rows << 'x' << columns
				<< " matrices having " << count << " byte elements" << std::endl;
			std::vector<uint8_t> matrices[2]; // beginning of vectors is 16-byte aligned on x64
			for (auto& matrix : matrices)
			{
				matrix.reserve(count);
				int cur;
				for (int i = 0; i < count; ++i)
				{
					ifs >> cur;
					if (cur < 0)
						std::cerr << "Value too low, assuming 0" << std::endl, cur = 0;
					else if (cur > 255)
						std::cerr << "Value too big, assuming 255" << std::endl, cur = 255;
					matrix.push_back(cur);
				}
			}

			if (!ifs)
			{
				std::cerr << "Error reading values from file" << std::endl;
				return 0;
			}

			std::cout << std::endl;
			bench("C++ dot product arbitrary byte values: ",
				mat_dot_prod_gen, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary byte values no extensions: ",
				mat_dot_prod_gen_masm_nommx, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary byte values mmx: ",
				mat_dot_prod_gen_masm_mmx, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary byte values sse 32-bit sum: ",
				mat_dot_prod_gen_masm_sse_32bitsum, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary byte values sse 16-bit sum: ",
				mat_dot_prod_gen_masm_sse_16bitsum, matrices[0].data(), matrices[1].data(), count);
		}
		else if (test == "2")
		{
			std::cout << "Dot-multiplying two " << rows << 'x' << columns
				<< " matrices having " << count << " float elements" << std::endl;
			std::vector<float> matrices[2]; // beginning of vectors is 16-byte aligned on x64
			for (auto& matrix : matrices)
			{
				matrix.assign(count, 0.f);
				for (int i = 0; i < count; ++i)
					ifs >> matrix[i];
			}

			if (!ifs)
			{
				std::cerr << "Error reading values from file" << std::endl;
				return 0;
			}

			std::cout << std::endl;
			bench("C++ dot product arbitrary float values: ",
				matf_dot_prod_gen, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary float values sse dpps: ",
				matf_dot_prod_gen_masm_sse, matrices[0].data(), matrices[1].data(), count);
			bench("ASM dot product arbitrary float values sse mulps: ",
				matf_dot_prod_gen_masm_sse_mulps, matrices[0].data(), matrices[1].data(), count);
		}
		else
			std::cout << "Unimplemented" << std::endl;

		std::cout << std::endl;
		std::cout << "Do you want to test another file? ('Y' or 'y' for yes): ";
		std::getline(std::cin, again);
	} while (again == "Y" || again == "y");

	return 0;
}