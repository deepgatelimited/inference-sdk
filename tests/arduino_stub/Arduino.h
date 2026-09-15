#pragma once
#include <cstdint>
#include <cmath>
#include <string>

// Host tests check the sketch logic. Test microphone capture on the board.
static uint32_t test_millis;
static int test_led = 1;
static constexpr int LED_BUILTIN = 0, PC9 = 9, PC8 = 8;
static constexpr int HIGH = 1, LOW = 0, OUTPUT = 1, INPUT = 0, CPU_78MHZ = 78;
static uint32_t millis() { return test_millis; }
static uint32_t micros() { return test_millis * 1000; }
static void delay(int ms) { test_millis += ms; }
static void pinMode(int, int) {}
static void digitalWrite(int pin, int level) { if (pin == LED_BUILTIN) test_led = level; }
static void setCPUClock(int) {}
static struct {
    std::string output;
    operator bool() const { return true; }
    void begin(int) {}
    void print(const char *s) { output += s; }
    void print(char c) { output += c; }
    template<class T> void print(T value) { output += std::to_string(value); }
    void println() { output += '\n'; }
    template<class T> void println(T value) { print(value); println(); }
} Serial;
