#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <vector>

using Matrix8x8 = const uint8_t[8][8];
using MatrixGen = const uint8_t*;
using DotMul64Fn = uint32_t(*)(Matrix8x8, Matrix8x8);
using DotMulGenFn = uint32_t(*)(MatrixGen, MatrixGen, int count);

uint32_t mat_dot_prod_64(Matrix8x8 a, Matrix8x8 b);
uint32_t mat_dot_prod_gen(MatrixGen a, MatrixGen b, int count);

extern "C"
{
	uint32_t mat_dot_prod_64_masm_nommx(Matrix8x8 a, Matrix8x8 b);
	uint32_t mat_dot_prod_64_masm_mmx(Matrix8x8 a, Matrix8x8 b);
	uint32_t mat_dot_prod_gen_masm_nommx(MatrixGen a, MatrixGen b, int count);
	uint32_t mat_dot_prod_gen_masm_mmx(MatrixGen a, MatrixGen b, int count);
}

static_assert(std::is_same_v<decltype(&mat_dot_prod_64), DotMul64Fn>);
static_assert(std::is_same_v<decltype(&mat_dot_prod_gen), DotMulGenFn>);

constexpr int BENCH_REPEATS = 1000000;

uint32_t mat_dot_prod_64(Matrix8x8 a, Matrix8x8 b)
{
	uint32_t sum = 0;
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j)
			sum += static_cast<uint32_t>(a[i][j])* b[i][j];
	return sum;
}

uint32_t mat_dot_prod_gen(MatrixGen a, MatrixGen b, int count)
{
	uint32_t sum = 0;
	for (int i = 0; i < count; ++i)
		sum += static_cast<uint32_t>(a[i])* b[i];
	return sum;
}

template<typename Fn, typename... Params>
void bench(std::string_view description,
	Fn mulFn, Params... params)
{
	using namespace std::chrono;
	
	auto start = high_resolution_clock::now();
	volatile uint32_t result;
	for (int i = 0; i < BENCH_REPEATS; ++i)
		result = mulFn(params...);
	auto end = high_resolution_clock::now();
	std::cout << description << std::endl;
	std::cout << "    Result: " << result << std::endl;
	std::cout << "    Time: " << duration_cast<milliseconds>(
		end - start).count() << " milliseconds" << std::endl;
}

int main()
{
	constexpr uint8_t a[8][8] = {
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 4, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 6, 0, 0, 0, 0, 0},
		{0, 2, 0, 0, 0, 0, 0, 0},
		{0, 0, 4, 0, 0, 5, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0}
	};
	constexpr uint8_t b[8][8] = {
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 2, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 8, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 3, 0, 0, 0, 0, 0},
		{0, 0, 5, 0, 0, 0, 0, 0},
		{7, 0, 0, 0, 0, 0, 9, 0},
		{0, 0, 0, 0, 0, 0, 0, 0}
	};

	std::cout << "All calculations are done "
		<< BENCH_REPEATS << " times" << std::endl;
	std::cout << std::endl;
	std::cout << "Dot-multiplying two 8x8 matrices having 64 elements" << std::endl;
	bench("C++ dot product 64 values: ",
		mat_dot_prod_64, a, b);
	bench("ASM dot product 64 values no mmx: ",
		mat_dot_prod_64_masm_nommx, a, b);
	bench("ASM dot product 64 values mmx: ",
		mat_dot_prod_64_masm_mmx, a, b);
	std::cout << std::endl;

	std::cout << "Enter filename for general matrices: ";
	std::string filename;
	do
	{
		if (!filename.empty())
			std::cout << "File does not exist, enter again please: ";
		std::getline(std::cin, filename);
	} while (!std::filesystem::exists(filename));
	
	std::ifstream ifs(filename);
	int rows, columns, cur;
	ifs >> rows >> columns;
	const int count = rows * columns;
	std::cout << "Dot-multiplying two " << rows << 'x' << columns
		<< " matrices having " << count << " elements" << std::endl;
	std::vector<uint8_t> matrices[2];
	for (auto& matrix : matrices)
	{
		matrix.reserve(count);
		for (int i = 0; i < count; ++i)
		{
			ifs >> cur;
			if (cur < 0 || 255 < cur)
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
	bench("C++ dot product arbitrary values: ",
		mat_dot_prod_gen, matrices[0].data(), matrices[1].data(), count);
	bench("ASM dot product arbitrary values no mmx: ",
		mat_dot_prod_gen_masm_nommx, matrices[0].data(), matrices[1].data(), count);
	bench("ASM dot product arbitrary values mmx: ",
		mat_dot_prod_gen_masm_mmx, matrices[0].data(), matrices[1].data(), count);

	return 0;
}