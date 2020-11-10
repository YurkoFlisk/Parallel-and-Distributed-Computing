#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <thread>
#include <bit>
#include <omp.h>
#include <intrin.h>

constexpr uint32_t BASE_BITS = 32;
constexpr uint64_t BASE = 1ull << BASE_BITS;
constexpr uint64_t BASE_MASK = BASE - 1;

constexpr uint32_t MAX_DIGIT_COUNT = 512;
constexpr uint32_t MAX_BIT_COUNT = MAX_DIGIT_COUNT * BASE_BITS;

constexpr int BENCH_REPEATS = 1;

enum ComparisonResult {
	COMP_LESS = -1, COMP_EQUAL, COMP_GREATER
};

struct LongNum;
ComparisonResult compare(const LongNum& a, const LongNum& b);

struct LongNum
{
	LongNum(uint32_t siz = 1, uint32_t num = 0)
		: digits(siz)
	{
		assert(siz);
		digits[0] = num;
	}
	LongNum(const std::vector<uint32_t>& digs)
		: digits(digs)
	{}
	LongNum(std::vector<uint32_t>&& digs)
		: digits(std::move(digs))
	{}

	uint32_t size() const
	{
		return digits.size();
	}
	bool isZero() const
	{
		return digits.size() == 1 && digits[0] == 0;
	}
	const uint32_t& operator[](uint32_t idx) const
	{
		assert(idx < size());
		return digits[idx];
	}
	uint32_t& operator[](uint32_t idx)
	{
		assert(idx < size());
		return digits[idx];
	}
	void removeTrailing0()
	{
		while (digits.size() > 1 && !digits.back())
			digits.pop_back();
	}
	void prependLowerDigit(uint32_t dig)
	{
		if (isZero())
			digits[0] = dig;
		else
			digits.insert(std::begin(digits), dig);
	}
	LongNum higherDigits(uint32_t cnt) const
	{
		assert(cnt <= size());
		LongNum ret(cnt);
		std::copy(std::cend(digits) - cnt, std::cend(digits),
			std::begin(ret.digits));
		return ret;
	}
	LongNum lowerDigits(uint32_t cnt) const
	{
		assert(cnt <= size());
		LongNum ret(cnt);
		std::copy(std::cbegin(digits), std::cbegin(digits) + cnt,
			std::begin(ret.digits));
		return ret;
	}
	bool testBit(uint32_t idx) const
	{
		assert(idx < BASE_BITS * size());
		return digits[idx / BASE_BITS] & (1u << (idx % BASE_BITS));
	}
	bool lowestBit() const
	{
		return digits[0] & 1;
	}
	LongNum& operator+=(uint32_t rhs)
	{
		uint64_t carry;
		const uint64_t newDig = digits[0] + (uint64_t)rhs;
		if (newDig >= BASE)
			digits[0] = newDig - BASE, carry = 1;
		else
			digits[0] = newDig, carry = 0;
		for (uint32_t i = 1; i < size(); ++i)
		{
			const uint64_t newDig = digits[i] + carry;
			if (newDig == BASE)
				digits[i] = 0, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		if (carry)
			digits.push_back(1);
		return *this;
	}
	LongNum& operator-=(uint32_t rhs)
	{
		int64_t carry;
		const int64_t newDig = digits[0] - (int64_t)rhs;
		if (newDig < 0)
			digits[0] = newDig + BASE, carry = 1;
		else
			digits[0] = newDig, carry = 0;
		for (uint32_t i = 1; i < size(); ++i)
		{
			const int64_t newDig = digits[i] - carry;
			if (newDig == -1)
				digits[i] = BASE - 1, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		if (carry)
			assert(false);
		removeTrailing0();
		return *this;
	}
	LongNum& operator+=(const LongNum& rhs)
	{
		if (size() < rhs.size())
			digits.resize(rhs.size());
		uint64_t carry = 0;
		uint32_t i = 0;
		for (; i < rhs.size(); ++i)
		{
			const uint64_t newDig = digits[i]
				+ (uint64_t)rhs.digits[i] + carry;
			if (newDig >= BASE)
				digits[i] = newDig - BASE, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		for (; i < size(); ++i)
		{
			const uint64_t newDig = digits[i] + carry;
			if (newDig == BASE)
				digits[i] = 0, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		if (carry)
			digits.push_back(1);
		return *this;
	}
	LongNum& operator-=(const LongNum& rhs)
	{
		assert(compare(rhs, *this) != COMP_GREATER);
		int64_t _t = rhs[0];
		int64_t carry = 0;
		uint32_t i = 0;
		for (; i < rhs.size(); ++i)
		{
			const int64_t newDig = digits[i]
				- (int64_t)rhs.digits[i] - carry;
			if (newDig < 0)
				digits[i] = newDig + BASE, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		for (; i < size(); ++i)
		{
			const int64_t newDig = digits[i] - carry;
			if (newDig == -1)
				digits[i] = BASE - 1, carry = 1;
			else
				digits[i] = newDig, carry = 0;
		}
		if (carry)
			assert(false);
		removeTrailing0();
		return *this;
	}

	std::vector<uint32_t> digits; // Little Endian, [0] is lowest
};

ComparisonResult compare(const LongNum& a, const LongNum& b)
{
	if (a.size() < b.size())
		return COMP_LESS;
	else if (b.size() < a.size())
		return COMP_GREATER;
	for (uint32_t i = a.size() - 1; i != (uint32_t)(-1); --i)
		if (a[i] < b[i])
			return COMP_LESS;
		else if (b[i] < a[i])
			return COMP_GREATER;
	return COMP_EQUAL;
}

bool operator==(const LongNum& a, const LongNum& b)
{
	return compare(a, b) == COMP_EQUAL;
}

bool operator<(const LongNum& a, const LongNum& b)
{
	return compare(a, b) == COMP_LESS;
}

bool operator>(const LongNum& a, const LongNum& b)
{
	return compare(a, b) == COMP_GREATER;
}

LongNum operator+(LongNum a, uint32_t b)
{
	return a += b;
}

LongNum operator-(LongNum a, uint32_t b)
{
	return a -= b;
}

LongNum operator+(LongNum a, const LongNum& b)
{
	return a += b;
}

LongNum operator-(LongNum a, const LongNum& b)
{
	return a -= b;
}

LongNum mul(const LongNum& a, uint32_t b)
{
	LongNum result(a.size() + 1);
	uint32_t carry = 0, i = 0;
	for (; i < a.size(); ++i)
	{
		const uint64_t newDigit = (uint64_t)a[i] * b + carry;
		result[i] = newDigit & BASE_MASK;
		carry = newDigit >> BASE_BITS;
	}
	result[i] = carry;
	result.removeTrailing0();
	return result;
}

LongNum mul(const LongNum& a, const LongNum& b)
{
	LongNum result(a.size() + b.size());
	for (uint32_t i = 0; i < a.size(); ++i)
	{
		uint32_t carry = 0, rIdx = i;
		for (uint32_t j = 0; j < b.size(); ++j, ++rIdx)
		{
			const uint64_t newDigit = result[rIdx]
				+ (uint64_t)a[i] * b[j] + carry;
			result[rIdx] = newDigit & BASE_MASK;
			carry = newDigit >> BASE_BITS;
		}
		result[rIdx] = carry;
	}
	result.removeTrailing0();
	return result;
}

// Returns {quotient, remainder}
std::pair<LongNum, LongNum> divmod(const LongNum& a, const LongNum& b)
{
	if (a.size() < b.size())
		return { LongNum(), a };

	std::vector<uint32_t> quoDigitsBE; // Big Endian, [0] is highest
	quoDigitsBE.reserve(a.size() - b.size() + 1);
	LongNum rem = a.higherDigits(b.size());

	auto reduce = [&]() {
		uint32_t digitLow = 0, digitHigh = BASE_MASK;
		while (digitHigh - digitLow)
		{
			const uint32_t mid = digitHigh - ((digitHigh - digitLow) >> 1);
			const LongNum test = mul(b, mid);
			if (auto comp = compare(test, rem); comp == COMP_LESS)
				digitLow = mid;
			else if (comp == COMP_GREATER)
				digitHigh = mid - 1;
			else // comp == COMP_EQUAL
				digitLow = digitHigh = mid;
		}
		rem -= mul(b, digitLow);
		quoDigitsBE.push_back(digitLow);
	};
	
	reduce();
	for (uint32_t i = a.size() - b.size() - 1; i != (uint32_t)(-1); --i)
	{
		rem.prependLowerDigit(a[i]);
		reduce();
	}

	LongNum quo(quoDigitsBE.size());
	std::copy(std::cbegin(quoDigitsBE), std::cend(quoDigitsBE),
		std::rbegin(quo.digits));
	quo.removeTrailing0();
	return { quo, rem };
}

LongNum operator/(const LongNum& a, const LongNum& b)
{
	return divmod(a, b).second;
}

LongNum operator%(const LongNum& a, const LongNum& b)
{
	return divmod(a, b).second;
}

LongNum fromString(std::string_view str)
{
	LongNum ret;
	for (char ch : str)
	{
		if (!isdigit(ch))
			throw std::invalid_argument(
				"'str' is not an unsigned integral number");
		const uint32_t dig = ch - '0';
		ret = mul(ret, 10) + dig;
	}
	return ret;
}

std::string toString(LongNum n)
{
	static const LongNum ten(1, 10);

	if (n.isZero())
		return "0";

	std::string result;
	LongNum rem;
	while (!n.isZero())
	{
		std::tie(n, rem) = divmod(n, ten);
		result.push_back('0' + rem.digits[0]);
	}

	std::reverse(std::begin(result), std::end(result));
	return result;
}

std::ostream& operator<<(std::ostream& ostr, const LongNum& a)
{
	return ostr << toString(a);
}

// Right-to-left binary algorithm
LongNum powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1);
	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		if (pow.testBit(idx))
			res = mul(res, x) % mod;
		x = mul(x, x) % mod;
	}
	return res;
}

// Left-to-right binary algorithm
LongNum ltrpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1);
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		res = mul(res, res) % mod;
		if (pow.testBit(idx))
			res = mul(res, x) % mod;
	}
	return res;
}

// Montgomery ladder algorithm
LongNum ladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum r[2] = { LongNum(1, 1), x }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		newR[curBit] = mul(r[curBit], r[curBit]) % mod;
		newR[!curBit] = mul(r[0], r[1]) % mod;

		r[0] = newR[0];
		r[1] = newR[1];
	}
	return r[0];
}

LongNum P_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1), x2;
	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				if (pow.testBit(idx))
					res = mul(res, x) % mod;
			}
			#pragma omp section
			{
				x2 = mul(x, x) % mod;
			}
		}

		x = x2;
	}
	return res;
}

LongNum T_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1), x2;
	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		std::thread t1([&]() {
			if (pow.testBit(idx))
				res = mul(res, x) % mod;
		});
		std::thread t2([&]() {
			x2 = mul(x, x) % mod;
		});
		t1.join();
		t2.join();

		x = x2;
	}
	return res;
}

LongNum D_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	// const uint32_t powHSize = pow.size() / 3;
	uint32_t powHSize = 0, bitCnt = 0, desiredHBitCnt = 0;
	for (uint32_t i = 0; i < pow.size(); ++i)
		desiredHBitCnt += __popcnt(pow[i]);
	desiredHBitCnt /= 3;
	while (true)
	{
		bitCnt += __popcnt(pow[pow.size() - 1 - powHSize]);
		if (bitCnt > desiredHBitCnt)
			break;
		++powHSize;
	}
	if (powHSize == 0)
		return powmod(x, pow, mod);

	const uint32_t powLSize = pow.size() - powHSize;
	const LongNum powH = pow.higherDigits(powHSize);
	const LongNum powL = pow.lowerDigits(powLSize);
	LongNum resH, resL;

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			LongNum xx = x;
			for (uint32_t i = 0; i < BASE_BITS * powLSize; ++i)
				xx = mul(xx, xx) % mod;
			resH = powmod(xx, powH, mod);
		}
		#pragma omp section
		{
			resL = powmod(x, powL, mod);
		}
	}

	return mul(resH, resL) % mod;
}

LongNum DT_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	// const uint32_t powHSize = pow.size() / 3;
	uint32_t powHSize = 0, bitCnt = 0, desiredHBitCnt = 0;
	for (uint32_t i = 0; i < pow.size(); ++i)
		desiredHBitCnt += __popcnt(pow[i]);
	desiredHBitCnt /= 3;
	while (true)
	{
		bitCnt += __popcnt(pow[pow.size() - 1 - powHSize]);
		if (bitCnt > desiredHBitCnt)
			break;
		++powHSize;
	}
	if (powHSize == 0)
		return powmod(x, pow, mod);

	const uint32_t powLSize = pow.size() - powHSize;
	const LongNum powH = pow.higherDigits(powHSize);
	const LongNum powL = pow.lowerDigits(powLSize);
	LongNum resH, resL;

	std::thread t1([&]() {
		LongNum xx = x;
		for (uint32_t i = 0; i < BASE_BITS * powLSize; ++i)
			xx = mul(xx, xx) % mod;
		resH = powmod(xx, powH, mod);
	});
	std::thread t2([&]() {
		resL = powmod(x, powL, mod);
	});
	t1.join();
	t2.join();

	return mul(resH, resL) % mod;
}

// Parallel Montgomery ladder algorithm
LongNum P_ladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum r[2] = { LongNum(1, 1), x }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				newR[curBit] = mul(r[curBit], r[curBit]) % mod;
			}
			#pragma omp section
			{
				newR[!curBit] = mul(r[0], r[1]) % mod;
			}
		}
		r[0] = newR[0];
		r[1] = newR[1];
	}
	return r[0];
}

// Parallel std::thread Montgomery ladder algorithm
LongNum T_ladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum r[2] = { LongNum(1, 1), x }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		std::thread t1([&]() {
			newR[curBit] = mul(r[curBit], r[curBit]) % mod;
		});
		std::thread t2([&]() {
			newR[!curBit] = mul(r[0], r[1]) % mod;
		});
		t1.join();
		t2.join();

		r[0] = newR[0];
		r[1] = newR[1];
	}
	return r[0];
}

template<typename Fn, typename... Params>
void bench(const std::vector<std::pair<
	std::string_view, Fn>>& fns, Params&&... params)
{
	using ResultType = std::invoke_result_t<Fn, Params...>;

	auto testFn = [](const Fn& fn, Params&&... params) {
		using namespace std::chrono;

		ResultType curRes;
		auto start = high_resolution_clock::now();
		for (int i = 0; i < BENCH_REPEATS; ++i)
			curRes = fn(std::forward<Params>(params)...);
		auto end = high_resolution_clock::now();
		return std::pair{ duration_cast<microseconds>(
			end - start).count(), curRes };
	};
	
	ResultType result;
	bool first = true;
	for (const auto& [desc, fn] : fns)
	{
		const auto& [dur, curRes] = testFn(fn, std::forward<Params>(params)...);
		std::cout << desc << std::endl;
		std::cout << "    Time: " << dur << " microseconds" << std::endl;
		if (first)
			result = curRes;
		else if (result != curRes)
		{
			std::cout << "Results disagree, stopping bench" << std::endl;
			std::cout << "Previous results were: " << std::endl;
			std::cout << "    " << result << std::endl;
			std::cout << "Current result is: " << std::endl;
			std::cout << "    " << curRes << std::endl;
			return;
		}
		first = false;
	}

	std::cout << "All results agree: " << result << std::endl;
}

int main()
{
	std::string aStr, bStr, mStr;
	std::cout << "Enter a (base): " << std::endl;
	std::cin >> aStr;
	std::cout << "Enter b (exponent): " << std::endl;
	std::cin >> bStr;
	std::cout << "Enter m (modulus): " << std::endl;
	std::cin >> mStr;
	const LongNum a = fromString(aStr), b = fromString(bStr),
		m = fromString(mStr);
	
	std::cout << "a: " << BASE_BITS * a.size() << " bits" << std::endl;
	std::cout << "b: " << BASE_BITS * b.size() << " bits" << std::endl;
	std::cout << "m: " << BASE_BITS * m.size() << " bits" << std::endl;

	/*std::cout << "a + b = " << toString(a + b) << std::endl;
	std::cout << "a - b = " << toString(a - b) << std::endl;
	std::cout << "a * b = " << toString(mul(a, b)) << std::endl;
	const auto [quot, rem] = divmod(a, b);
	std::cout << "a / b = " << toString(quot) << std::endl;
	std::cout << "a % b = " << toString(rem) << std::endl;
	std::cout << "a^b mod m = " << toString(P_powmod(a, b, m)) << std::endl;
	std::cout << "a^b mod m = " << toString(powmod(a, b, m)) << std::endl;*/

	using namespace std::string_view_literals;
	std::vector<std::pair<std::string_view, decltype(&powmod)>> funcs{
		{"Binary (right-to-left) power mod: "sv, powmod},
		{"OpenMP parallel binary power mod: "sv, P_powmod},
		{"std::thread parallel binary power mod: "sv, T_powmod},
		{"Dividing OpenMP parallel binary power mod: "sv, D_powmod},
		{"Dividing std::thread parallel binary power mod: "sv, DT_powmod},

		{"Left-to-right binary power mod: "sv, ltrpowmod},
		{"Montgomery ladder: "sv, ladpowmod},
		{"OpenMP parallel montgomery ladder: "sv, P_ladpowmod},
		{"std::thread parallel montgomery ladder: "sv, T_ladpowmod}
	};
	bench(funcs, a, b, m);

	return 0;
}