#pragma once
// Coda SPSC (single-producer/single-consumer) lock-free a dimensione fissa (punto 7
// fase 2, sotto-punto d): un solo thread scrive (push), un solo thread legge (pop) --
// esattamente lo schema richiesto da runtime_architettura_realtime.md ("mai lock nel
// thread audio"). Capacita' utile = Capacity-1 (uno slot sempre vuoto per distinguere
// pieno da vuoto senza un contatore separato -- classico ring buffer a due indici).
#include <array>
#include <atomic>
#include <cstddef>

namespace phimo {

template <typename T, std::size_t Capacity>
class SpscQueue {
public:
    // Chiamato SOLO dal thread produttore. false se piena (il chiamante decide: scarta,
    // best-effort -- stesso spirito di Python, mai un blocco nel thread audio).
    bool push(const T& item) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) % Capacity;
        if (next == tail_.load(std::memory_order_acquire)) return false;
        buf_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Chiamato SOLO dal thread consumatore. false se vuota.
    bool pop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;
        out = buf_[tail];
        tail_.store((tail + 1) % Capacity, std::memory_order_release);
        return true;
    }

private:
    std::array<T, Capacity> buf_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

} // namespace phimo
