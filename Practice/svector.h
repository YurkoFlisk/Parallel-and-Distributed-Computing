#pragma once

#include <cassert>
#include <iterator>

template<typename T, size_t Capacity>
class SVector
{
public:
	using size_type = size_t;
	using value_type = T;
	using iterator = T*;
	using const_iterator = const T*;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	constexpr SVector(size_t siz = 0)
		: siz(siz)
	{
		assert(siz <= Capacity);
	}
	template<size_t RhsCapacity>
	constexpr SVector(const SVector<T, RhsCapacity>& rhs)
		: siz(rhs.siz)
	{
		assert(siz <= Capacity);
		std::copy_n(rhs.begin(), siz, begin());
	}

	constexpr bool empty() const noexcept { return siz == 0; }
	constexpr size_t size() const noexcept { return siz; }
	constexpr void clear() noexcept { siz = 0; }
	constexpr void resize(size_t newSize)
	{
		assert(newSize <= Capacity);
		for (size_t i = siz; i < newSize; ++i)
			data[i] = 0;
		siz = newSize;
	}

	constexpr void push_back(const T& elem)
	{
		assert(siz < Capacity);
		data[siz++] = elem;
	}
	constexpr void push_back(T&& elem)
	{
		assert(siz < Capacity);
		data[siz++] = std::move(elem);
	}
	constexpr void pop_back()
	{
		assert(siz > 0);
		--siz;
	}
	constexpr T& back()
	{
		assert(siz > 0);
		return data[siz - 1];
	}
	constexpr const T& back() const
	{
		assert(siz > 0);
		return data[siz - 1];
	}

	constexpr iterator insert(iterator place, const T& elem)
	{
		assert(siz < Capacity);
		assert(begin() <= place && place <= end());
		for (iterator it = end(); it > place; --it)
			*it = *(it - 1);
		*place = elem;
		++siz;
		return place;
	}

	constexpr iterator insert(iterator place, T&& elem)
	{
		assert(siz < Capacity);
		assert(begin() <= place && place <= end());
		for (iterator it = end(); it > place; --it)
			*it = *(it - 1);
		*place = std::move(elem);
		++siz;
		return place;
	}

	constexpr iterator insert(iterator place, size_t cnt, const T& elem)
	{
		assert(siz + cnt <= Capacity);
		assert(begin() <= place && place <= end());
		for (iterator it = end() + cnt - 1u; it >= place + cnt; --it)
			*it = *(it - cnt);
		for (size_t i = 0; i < cnt; ++i)
			*(place + i) = elem;
		siz += cnt;
		return place;
	}

	constexpr iterator insert(iterator place, size_t cnt, T&& elem)
	{
		assert(siz + cnt <= Capacity);
		assert(begin() <= place && place <= end());
		for (iterator it = end() + cnt - 1u; it >= place + cnt; --it)
			*it = *(it - cnt);
		for (size_t i = 0; i < cnt; ++i)
			*(place + i) = std::move(elem);
		siz += cnt;
		return place;
	}

	constexpr void erase(iterator place, size_t cnt)
	{
		// assert(siz >= cnt);
		assert(begin() <= place && place <= end());
		if (siz <= cnt)
		{
			clear();
			return;
		}
		for (iterator it = begin(); it < end() - cnt; ++it)
			*it = *(it + cnt);
		siz -= cnt;
	}

	template<size_t RhsCapacity>
	constexpr T& operator=(const SVector<T, RhsCapacity>& rhs)
	{
		assert(rhs.siz <= Capacity);
		siz = rhs.siz;
		std::copy_n(rhs.begin(), siz, begin());
	}

	constexpr T& operator[](size_t idx) { return data[idx]; }
	constexpr const T& operator[](size_t idx) const { return data[idx]; }

	constexpr iterator begin() noexcept { return data; }
	constexpr iterator end() noexcept { return data + siz; }
	constexpr const_iterator begin() const noexcept { return data; }
	constexpr const_iterator end() const noexcept { return data + siz; }
	constexpr const_iterator cbegin() const noexcept { return data; }
	constexpr const_iterator cend() const noexcept { return data + siz; }
	constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	constexpr const_reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
	constexpr const_reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
	constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }
private:
	size_t siz;
	T data[Capacity];
};