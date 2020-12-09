#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <ranges>
#include <span>
#include <random>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <thread>
#include <bit>
#include <barrier>
#include <omp.h>

#include "svector.h"

constexpr uint32_t BASE_BITS = 32;
constexpr uint64_t BASE = 1ull << BASE_BITS;
constexpr uint64_t BASE_MASK = BASE - 1;

constexpr uint32_t MAX_DIGIT_COUNT = 400;
constexpr uint32_t MAX_BIT_COUNT = MAX_DIGIT_COUNT * BASE_BITS;

constexpr int BENCH_REPEATS = 1;

enum ComparisonResult {
	COMP_LESS = -1, COMP_EQUAL, COMP_GREATER
};

struct LongNum;
ComparisonResult compare(const LongNum& a, const LongNum& b);

struct LongNum
{
	using DigitContainer = SVector<uint32_t, MAX_DIGIT_COUNT>;

	LongNum(uint32_t siz = 1, uint32_t num = 0)
		: digits(siz)
	{
		assert(siz);
		for (uint32_t i = 0; i < siz; ++i)
			digits[i] = num;
	}
	LongNum(const DigitContainer& digs)
		: digits(digs)
	{}
	LongNum(DigitContainer&& digs)
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
		if (cnt == 0)
			return LongNum(1, 0);
		assert(cnt <= size());
		LongNum ret(cnt);
		std::copy(std::cend(digits) - cnt, std::cend(digits),
			std::begin(ret.digits));
		return ret;
	}
	LongNum lowerDigits(uint32_t cnt) const
	{
		//assert(cnt <= size());
		cnt = std::min(cnt, size());
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
	uint32_t popcount() const
	{
		uint32_t popcnt = 0;
		for (uint32_t i = 0; i < size(); ++i)
			popcnt += std::popcount(digits[i]);
		return popcnt;
	}
	LongNum shiftWordsUp(uint32_t cnt) const
	{
		LongNum res = *this;
		res.digits.insert(std::begin(res.digits), cnt, 0u);
		return res;
	}
	LongNum shiftWordsDown(uint32_t cnt) const
	{
		LongNum res = *this;
		res.digits.erase(std::begin(res.digits), cnt);
		if (res.digits.empty())
			res.digits.push_back(0);
		return res;
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

	DigitContainer digits; // Little Endian, [0] is lowest
};

uint32_t R_DIGIT_COUNT = 1; // R = pow(BASE, R_DIGIT_COUNT)
LongNum R;

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

bool operator<=(const LongNum& a, const LongNum& b)
{
	return compare(a, b) != COMP_GREATER;
}

bool operator>=(const LongNum& a, const LongNum& b)
{
	return compare(a, b) != COMP_LESS;
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

LongNum uncheckedMinus(LongNum a, const LongNum& b)
{
	// return a.uncheckedMinusEq(b);
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
	LongNum result(a.size() + b.size(), 0);
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

	LongNum::DigitContainer quoDigitsBE; // Big Endian, [0] is highest
	//quoDigitsBE.reserve(a.size() - b.size() + 1);
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

int64_t modinv(int64_t a, int64_t n)
{
	int64_t t = 0, r = n, newT = 1, newR = a;

	while (newR)
	{
		const auto [quot, rem] = std::div(r, newR);
		std::tie(t, newT) = std::tuple(newT, t - quot * newT);
		std::tie(r, newR) = std::tuple(newR, rem);
		newT %= n;
	}

	assert(r == 1);
	if (t >= n)
		return t - n;
	return t;
}

// a^(-1) mod (BASE^k)
// According to algorithm from https://eprint.iacr.org/2017/411.pdf
// Modified by doing first iteration separately and holding '-b' from
// the paper in variable b (because for iterations with i >= 1 'b' from paper
// is non-positive, so here b is non-negative as is needed for unsigned LongNum type) 
LongNum modinv_base_k(const LongNum& a, const uint32_t k)
{
	const uint32_t c = modinv(a[0], BASE);
	LongNum::DigitContainer x;

	x.push_back(c);
	LongNum b = (mul(a, c) - 1u).shiftWordsDown(1);
	for (uint32_t i = 1; i < k; ++i)
	{
		const uint32_t cur = (~b[0] + 1) * c;
		x.push_back(cur);

		b = b + mul(a, cur);
		b = b.shiftWordsDown(1); // b is divisible by BASE here
	}

	return x;
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

// Montgomery reduction
LongNum redc(const LongNum& x, const LongNum& mod, const LongNum& modRInv)
{
	const LongNum m = mul(x.lowerDigits(R_DIGIT_COUNT), modRInv)
		.lowerDigits(R_DIGIT_COUNT);
	const LongNum t = (x + mul(m, mod))
		.shiftWordsDown(R_DIGIT_COUNT);
	if (t >= mod)
		return t - mod;
	return t;
}

// Binary algorithm with montgomery multiplication
LongNum monpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum xMon = x.shiftWordsUp(R_DIGIT_COUNT) % mod;
	LongNum resMon = R % mod;
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		if (pow.testBit(idx))
		{
			resMon = mul(resMon, xMon);
			resMon = redc(resMon, mod, modRInv);
		}
		xMon = mul(xMon, xMon);
		xMon = redc(xMon, mod, modRInv);
	}
	return redc(resMon, mod, modRInv);
}

// Binary left-to-right algorithm with montgomery multiplication
LongNum monltrpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum xMon = x.shiftWordsUp(R_DIGIT_COUNT) % mod;
	LongNum resMon = R % mod;
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		resMon = mul(resMon, resMon);
		resMon = redc(resMon, mod, modRInv);
		if (pow.testBit(idx))
		{
			resMon = mul(resMon, xMon);
			resMon = redc(resMon, mod, modRInv);
		}
	}

	return redc(resMon, mod, modRInv);
}

// Montgomery ladder algorithm with montgomery multiplication
LongNum monladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	LongNum r[2] = { R % mod, x.shiftWordsUp(R_DIGIT_COUNT) % mod }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		newR[curBit] = mul(r[curBit], r[curBit]);
		newR[curBit] = redc(newR[curBit], mod, modRInv);
		newR[!curBit] = mul(r[0], r[1]);
		newR[!curBit] = redc(newR[!curBit], mod, modRInv);

		r[0] = newR[0];
		r[1] = newR[1];
	}
	return redc(r[0], mod, modRInv);
}

// OpenMP-parallel right-to-left binary algorithm
LongNum P_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1), x2;
	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		#pragma omp parallel sections num_threads(2)
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

// std::thread-parallel right-to-left binary algorithm
LongNum T_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum res(1, 1), x2;

	std::barrier bar(2);
	std::thread t1([&]() {
		for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
		{
			if (pow.testBit(idx))
				res = mul(res, x) % mod;
			bar.arrive_and_wait();
			x = x2;
			bar.arrive_and_wait();
		}
	});
	std::thread t2([&]() {
		for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
		{
			x2 = mul(x, x) % mod;
			bar.arrive_and_wait();
			bar.arrive_and_wait();
		}
	});

	t1.join();
	t2.join();

	return res;
}

// OpenMP-parallel split exponent right-to-left binary algorithm
LongNum SP_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const uint32_t powHSize = pow.size() / 4;
	/*uint32_t powHSize = 0, bitCnt = 0, desiredHBitCnt = 0;
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
		return powmod(x, pow, mod);*/

	const uint32_t powLSize = pow.size() - powHSize;
	const LongNum powH = pow.higherDigits(powHSize);
	const LongNum powL = pow.lowerDigits(powLSize);
	LongNum resH, resL;

	#pragma omp parallel sections num_threads(2)
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

// std::thread-parallel split exponent right-to-left binary algorithm
LongNum ST_powmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const uint32_t powHSize = pow.size() / 4;
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

// OpenMP-parallel split exponent left-to-right binary algorithm
LongNum SP_ltrpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const uint32_t powHSize = pow.size() / 4;
	const uint32_t powLSize = pow.size() - powHSize;
	const LongNum powH = pow.higherDigits(powHSize);
	const LongNum powL = pow.lowerDigits(powLSize);
	LongNum resH, resL;

	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{
			LongNum xx = x;
			for (uint32_t i = 0; i < BASE_BITS * powLSize; ++i)
				xx = mul(xx, xx) % mod;
			resH = ltrpowmod(xx, powH, mod);
		}
		#pragma omp section
		{
			resL = ltrpowmod(x, powL, mod);
		}
	}

	return mul(resH, resL) % mod;
}

// std::thread-parallel split exponent left-to-right binary algorithm
LongNum ST_ltrpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const uint32_t powHSize = pow.size() / 4;
	const uint32_t powLSize = pow.size() - powHSize;
	const LongNum powH = pow.higherDigits(powHSize);
	const LongNum powL = pow.lowerDigits(powLSize);
	LongNum resH, resL;

	std::thread t1([&]() {
		LongNum xx = x;
		for (uint32_t i = 0; i < BASE_BITS * powLSize; ++i)
			xx = mul(xx, xx) % mod;
		resH = ltrpowmod(xx, powH, mod);
	});
	std::thread t2([&]() {
		resL = ltrpowmod(x, powL, mod);
	});
	t1.join();
	t2.join();

	return mul(resH, resL) % mod;
}

// Parallel OpenMP Montgomery ladder algorithm
LongNum P_ladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum r[2] = { LongNum(1, 1), x }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		#pragma omp parallel sections num_threads(2)
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

	std::barrier bar(2);
	std::thread t1([&]() {
		for (uint32_t idx = BASE_BITS * pow.size() - 1;
			idx != (uint32_t)(-1); --idx)
		{
			const uint32_t curBit = pow.testBit(idx);
			newR[curBit] = mul(r[curBit], r[curBit]) % mod;
			bar.arrive_and_wait();
			r[0] = newR[0];
			bar.arrive_and_wait();
		}
	});
	std::thread t2([&]() {
		for (uint32_t idx = BASE_BITS * pow.size() - 1;
			idx != (uint32_t)(-1); --idx)
		{
			const uint32_t curBit = pow.testBit(idx);
			newR[!curBit] = mul(r[0], r[1]) % mod;
			bar.arrive_and_wait();
			r[1] = newR[1];
			bar.arrive_and_wait();
		}
	});

	t1.join();
	t2.join();

	return r[0];
}

// Parallel OpenMP binary right-to-left with Montgomery multiplication
LongNum P_monpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum xMon = x.shiftWordsUp(R_DIGIT_COUNT) % mod, xMon2;
	LongNum resMon = R % mod;
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
	{
		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
			{
				if (pow.testBit(idx))
				{
					resMon = mul(resMon, xMon);
					resMon = redc(resMon, mod, modRInv);
				}
			}
			#pragma omp section
			{
				xMon2 = mul(xMon, xMon);
				xMon2 = redc(xMon2, mod, modRInv);
			}
		}

		xMon = xMon2;
	}
	return redc(resMon, mod, modRInv);
}

// Parallel std::thread binary right-to-left with Montgomery multiplication
LongNum T_monpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	LongNum xMon = x.shiftWordsUp(R_DIGIT_COUNT) % mod, xMon2;
	LongNum resMon = R % mod;
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	std::barrier bar(2);
	std::thread t1([&]() {
		for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
		{
			if (pow.testBit(idx))
			{
				resMon = mul(resMon, xMon);
				resMon = redc(resMon, mod, modRInv);
			}
			bar.arrive_and_wait();
			xMon = xMon2;
			bar.arrive_and_wait();
		}
	});
	std::thread t2([&]() {
		for (uint32_t idx = 0; idx < BASE_BITS * pow.size(); ++idx)
		{
			xMon2 = mul(xMon, xMon);
			xMon2 = redc(xMon2, mod, modRInv);
			bar.arrive_and_wait();
			bar.arrive_and_wait();
		}
	});

	t1.join();
	t2.join();

	return redc(resMon, mod, modRInv);
}

// Parallel OpenMP Montgomery ladder algorithm with Montgomery multiplication
LongNum P_monladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	LongNum r[2] = { R % mod, x.shiftWordsUp(R_DIGIT_COUNT) % mod }, newR[2];
	for (uint32_t idx = BASE_BITS * pow.size() - 1;
		idx != (uint32_t)(-1); --idx)
	{
		const uint32_t curBit = pow.testBit(idx);
		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
			{
				newR[curBit] = mul(r[curBit], r[curBit]);
				newR[curBit] = redc(newR[curBit], mod, modRInv);
			}
			#pragma omp section
			{
				newR[!curBit] = mul(r[0], r[1]);
				newR[!curBit] = redc(newR[!curBit], mod, modRInv);
			}
		}
		r[0] = newR[0];
		r[1] = newR[1];
	}
	return redc(r[0], mod, modRInv);
}

// Parallel std::thread Montgomery ladder algorithm with Montgomery multiplication
LongNum T_monladpowmod(LongNum x, const LongNum& pow, const LongNum& mod)
{
	const LongNum modRInv = R - modinv_base_k(mod, R_DIGIT_COUNT);

	LongNum r[2] = { R % mod, x.shiftWordsUp(R_DIGIT_COUNT) % mod }, newR[2];
	std::barrier bar(2);
	std::thread t1([&]() {
		for (uint32_t idx = BASE_BITS * pow.size() - 1;
			idx != (uint32_t)(-1); --idx)
		{
			const uint32_t curBit = pow.testBit(idx);
			newR[curBit] = mul(r[curBit], r[curBit]);
			newR[curBit] = redc(newR[curBit], mod, modRInv);
			bar.arrive_and_wait();
			r[0] = newR[0];
			bar.arrive_and_wait();
		}
	});
	std::thread t2([&]() {
		for (uint32_t idx = BASE_BITS * pow.size() - 1;
			idx != (uint32_t)(-1); --idx)
		{
			const uint32_t curBit = pow.testBit(idx);
			newR[!curBit] = mul(r[0], r[1]);
			newR[!curBit] = redc(newR[!curBit], mod, modRInv);
			bar.arrive_and_wait();
			r[1] = newR[1];
			bar.arrive_and_wait();
		}
	});

	t1.join();
	t2.join();

	return redc(r[0], mod, modRInv);
}

using ModExpFnType = decltype(&powmod);

struct Testcase
{
	LongNum a;
	LongNum b;
	LongNum m;
};

struct Algorithm
{
	std::string_view desc;
	std::string_view name;
	ModExpFnType fn;
};

// Generic bench function, outputs results directly to cout
// Maybe will be removed later because of benchBrief
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
			curRes = fn(params...);
		auto end = high_resolution_clock::now();
		return std::pair{ duration_cast<microseconds>(
			end - start).count() / BENCH_REPEATS, curRes };
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
			std::cerr << "Results disagree, stopping bench" << std::endl;
			std::cerr << "Previous results were: " << std::endl;
			std::cerr << "    " << result << std::endl;
			std::cerr << "Current result is: " << std::endl;
			std::cerr << "    " << curRes << std::endl;
			return;
		}
		first = false;
	}

	std::cout << "All results agree: " << result << std::endl;
}

// Bench function for modular exponentiation algorithms
// Not generic, but outputs only average execution
// times (as a return value) and doesnt use cout/cerr
// Maybe will eventually fully replace 'bench' function here
std::vector<float> benchBrief(std::span<const Algorithm> fns,
	const Testcase& testcase)
{
	auto testFn = [&testcase](const auto& fn, auto&&... params) {
		using namespace std::chrono;

		LongNum curRes;
		auto start = high_resolution_clock::now();
		for (int i = 0; i < BENCH_REPEATS; ++i)
			curRes = fn(testcase.a, testcase.b, testcase.m);
		auto end = high_resolution_clock::now();
		return std::pair{ duration_cast<microseconds>(
			end - start).count() / (1000.f * BENCH_REPEATS), curRes };
	};
	
	std::vector<float> durs;
	LongNum result;
	bool first = true;
	for (const auto& [desc, name, fn] : fns)
	{
		const auto& [dur, curRes] = testFn(fn, testcase);
		durs.push_back(dur);
		if (first)
			result = curRes;
		else if (result != curRes)
		{
			std::stringstream ss;
			ss << "Results disagree, stopping bench\n";
			ss << "Previous results were: \n";
			ss << "    " << result << '\n';
			ss << "Current algorithm (" << name << " - "
				<< desc << ")  result is: \n";
			ss << "    " << curRes << '\n';
			throw std::runtime_error(ss.str());
		}
		first = false;
	}

	return durs;
}

const std::array algorithms = {
	Algorithm{"Binary (right-to-left) power mod: ", "powmod", powmod},
	Algorithm{"OpenMP parallel binary power mod: ", "P_powmod", P_powmod},
	Algorithm{"std::thread parallel binary power mod: ", "T_powmod", T_powmod},
	Algorithm{"Split OpenMP parallel binary power mod: ", "SP_powmod", SP_powmod},
	Algorithm{"Split std::thread parallel binary power mod: ", "ST_powmod", ST_powmod},

	Algorithm{"Left-to-right binary power mod: ", "ltrpowmod", ltrpowmod},
	Algorithm{"Split OpenMP parallel LTR binary power mod: ", "SP_ltrpowmod", SP_ltrpowmod},
	Algorithm{"Split std::thread parallel LTR binary power mod: ", "ST_ltrpowmod", ST_ltrpowmod},

	Algorithm{"Montgomery ladder: ", "ladpowmod", ladpowmod},
	Algorithm{"OpenMP parallel montgomery ladder: ", "P_ladpowmod", P_ladpowmod},
	Algorithm{"std::thread parallel montgomery ladder: ", "T_ladpowmod", T_ladpowmod},

	Algorithm{"Binary with Montgomery multiplication: ", "monpowmod", monpowmod},
	Algorithm{"OpenMP parallel binary power mod with Montgomery multiplication: ", "P_monpowmod", P_monpowmod},
	Algorithm{"std::thread parallel binary power mod with Montgomery multiplication: ", "T_monpowmod", T_monpowmod},

	Algorithm{"Left-to-right binary with Montgomery multiplication: ", "monltrpowmod", monltrpowmod},

	Algorithm{"Montgomery ladder with Montgomery multiplication: ", "monladpowmod", monladpowmod},
	Algorithm{"OpenMP parallel Montgomery ladder with Montgomery multiplication", "P_monladpowmod", P_monladpowmod},
	Algorithm{"std::thread parallel Montgomery ladder with Montgomery multiplication", "T_monladpowmod", T_monladpowmod}
};

void processTestcases(std::istream& in, std::ostream& out)
{
	std::vector<std::vector<float>> results(algorithms.size()); // ms
	std::vector<Testcase> testcases;

	std::string aStr, bStr, mStr;
	while (in >> aStr >> bStr >> mStr)
	{
		const LongNum a = fromString(aStr), b = fromString(bStr),
			m = fromString(mStr);
		if (!m.lowestBit()) // if m is odd
		{
			std::cerr << "Modulus should be odd for testing all algorithms" << std::endl;
			return;
		}
		testcases.emplace_back(a, b, m);
	}

	try
	{
		for (size_t testcaseIdx = 0; testcaseIdx < testcases.size(); ++testcaseIdx)
		{
			const Testcase& testcase = testcases[testcaseIdx];

			R_DIGIT_COUNT = testcase.m.size();
			R = LongNum(1, 1).shiftWordsUp(R_DIGIT_COUNT);
			// Now R is the smallest power of BASE that is greater than M

			std::cout << "Processing testcase #" << testcaseIdx << "...";
			auto curResults = benchBrief(algorithms, testcase);
			for (size_t i = 0; i < algorithms.size(); ++i)
				results[i].push_back(curResults[i]);
			std::cout << " done" << std::endl;
		}
	}
	catch (const std::runtime_error& err)
	{
		out << err.what();
		std::cerr << err.what();
		return;
	}

	std::cout << "Finished processing testcases.\nGenerating output..." << std::endl;
	for (size_t i = 0; i < algorithms.size(); ++i)
		out << algorithms[i].name << " - " << algorithms[i].desc << std::endl;
	out << std::endl;
	out << std::fixed << std::setprecision(2);
	out << "bits\t";
	for (const auto& testcase : testcases)
		out << BASE_BITS * testcase.m.size() << '\t';
	out << std::endl;
	for (size_t i = 0; i < algorithms.size(); ++i)
	{
		out << algorithms[i].name << '\t';
		for (float res : results[i])
			out << res << '\t';
		out << std::endl;
	}
	std::cout << " done" << std::endl;
}

void generateTestcases(std::ostream& out)
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_int_distribution<uint32_t> dist;
	auto genU32 = [&]() {return dist(engine); };
	auto generateLongNum = [&](uint32_t siz) {
		LongNum res(siz, 0);
		std::generate(res.digits.begin(), res.digits.end(), genU32);
		while (!res.lowestBit()) // Avoid even modulus
			res.digits[0] = genU32();
		while (!res.digits.back()) // Avoid zero highest digit
			res.digits.back() = genU32();
		return res;
	};

	std::cout << "Generating long numbers...";
	// Order of numbers in testcases - base (a), exponent (b), modulo (m)
	for (size_t i = 16; i < MAX_DIGIT_COUNT / 2; i += 16)
		out << generateLongNum(i) << '\n' << generateLongNum(i) << '\n'
			<< generateLongNum(i) << '\n' << std::endl;
	std::cout << " done" << std::endl;
}

void showHelp(std::string_view command)
{
	std::cout << "Valid modes of operation:\n";
	std::cout << command << " - interactive mode\n";
	std::cout << command << " gen <path> - generate and save test data to <path>\n";
	std::cout << command << " test <in> <out> - launch benchmark for tests in "
		"<in> and output results to <out>" << std::endl;
}

int main(int argc, char** argv)
{
	if (argc == 4)
	{
		if (strcmp(argv[1], "test") == 0)
		{
			std::ifstream in(argv[2]);
			std::ofstream out(argv[3]);
			processTestcases(in, out);
			in.close();
			out.close();
		}
		else
			showHelp(argv[0]);
		return 0;
	}
	else if (argc == 3)
	{
		if (strcmp(argv[1], "gen") == 0)
		{
			std::ofstream out(argv[2]);
			generateTestcases(out);
			out.close();
		}
		else
			showHelp(argv[0]);
		return 0;
	}
	else if (argc != 1)
	{
		showHelp(argv[0]);
		return 0;
	}

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
	std::cout << "b set bit count (popcount): " << b.popcount() << std::endl;
	std::cout << "m: " << BASE_BITS * m.size() << " bits" << std::endl;

	/*std::cout << "a + b = " << a + b << std::endl;
	std::cout << "a - b = " << a - b << std::endl;
	std::cout << "a * b = " << mul(a, b) << std::endl;
	const auto [quot, rem] = divmod(a, b);
	std::cout << "a / b = " << quot << std::endl;
	std::cout << "a % b = " << rem << std::endl;
	std::cout << "a^b mod m = " << P_powmod(a, b, m) << std::endl;
	std::cout << "a^b mod m = " << powmod(a, b, m) << std::endl;
	std::cout << "m^(-1) mod R = " << modinv_base_k(m, R_DIGIT_COUNT) << std::endl;*/

	using namespace std::string_view_literals;
	std::vector<std::pair<std::string_view, ModExpFnType>> funcs{
		{"Binary (right-to-left) power mod: "sv, powmod},
		{"OpenMP parallel binary power mod: "sv, P_powmod},
		{"std::thread parallel binary power mod: "sv, T_powmod},
		{"Split OpenMP parallel binary power mod: "sv, SP_powmod},
		{"Split std::thread parallel binary power mod: "sv, ST_powmod},

		{"Left-to-right binary power mod: "sv, ltrpowmod},
		{"Split OpenMP parallel LTR binary power mod: "sv, SP_ltrpowmod},
		{"Split std::thread parallel LTR binary power mod: "sv, ST_ltrpowmod},

		{"Montgomery ladder: "sv, ladpowmod},
		{"OpenMP parallel montgomery ladder: "sv, P_ladpowmod},
		{"std::thread parallel montgomery ladder: "sv, T_ladpowmod},
	};
	if (m.lowestBit()) // if m is odd
	{
		R_DIGIT_COUNT = m.size();
		R = LongNum(1, 1).shiftWordsUp(R_DIGIT_COUNT);
		// Now R is the smallest power of BASE that is greater than M
		funcs.insert(funcs.end(), {
			{"Binary with Montgomery multiplication: "sv, monpowmod},
			{"Left-to-right binary with Montgomery multiplication: "sv, monltrpowmod},
			{"Montgomery ladder with Montgomery multiplication: "sv, monladpowmod},
			{"OpenMP parallel binary power mod with Montgomery multiplication: "sv, P_monpowmod},
			{"std::thread parallel binary power mod with Montgomery multiplication: "sv, T_monpowmod},
			{"OpenMP parallel Montgomery ladder with Montgomery multiplication"sv, P_monladpowmod},
			{"std::thread parallel Montgomery ladder with Montgomery multiplication"sv, T_monladpowmod}
			});
	}
	else
		std::cerr << "Warning, algorithms based on Montgomery multiplication will"
		" be excluded because modulus is even!" << std::endl;

	bench(funcs, a, b, m);

	return 0;
}