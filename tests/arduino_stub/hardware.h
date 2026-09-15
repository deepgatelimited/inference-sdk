#pragma once
#include <cstdint>

struct LDMA_Descriptor_t { struct { int size, doneIfs; } xfer; };
struct LDMA_TransferCfg_t {};
struct IADC_Init_t { int warmup, srcClkPrescale; };
struct IADC_AllConfigs_t {
    struct { int reference, vRef, analogGain, osrHighSpeed, adcClkPrescale; } configs[1];
};
struct IADC_InitScan_t { int triggerSelect, triggerAction, dataValidLevel; bool fifoDmaWakeup; };
struct IADC_ScanTable_t { struct { int posInput, negInput; bool includeInScan; } entries[1]; };
struct TIMER_Init_TypeDef { bool enable; };
static struct { uint32_t SCANFIFODATA; } test_iadc;
static struct { uint32_t TOPB; } test_timer;
static struct { uint32_t CDBUSALLOC; } test_gpio;
#define IADC0 (&test_iadc)
#define TIMER0 (&test_timer)
#define GPIO (&test_gpio)
#define IADC_INIT_DEFAULT {}
#define IADC_ALLCONFIGS_DEFAULT {}
#define IADC_INITSCAN_DEFAULT {}
#define IADC_SCANTABLE_DEFAULT {}
#define TIMER_INIT_DEFAULT {}
#define LDMA_DESCRIPTOR_LINKREL_P2M_WORD(...) {}
#define LDMA_DESCRIPTOR_LINKREL_M2P_BYTE(...) {}
#define LDMA_TRANSFER_CFG_PERIPHERAL(...) {}
enum {
    cmuClock_EM01GRPACLK, cmuSelect_HFRCODPLL, cmuClock_IADC0, cmuClock_GPIO, cmuClock_TIMER0, cmuClock_PRS,
    cmuClock_IADCCLK, cmuSelect_EM01GRPACLK, iadcWarmupKeepWarm,
    iadcCfgReferenceInt1V2, iadcCfgAnalogGain1x, iadcCfgOsrHighSpeed32x,
    iadcCfgModeNormal, GPIO_CDBUSALLOC_CDODD0_ADC0, iadcTriggerSelPrs0PosEdge,
    iadcTriggerActionOnce, iadcFifoCfgDvl1, iadcPosInputPortCPin9,
    iadcNegInputGnd, iadcCmdStartScan, prsTypeAsync, prsSignalTIMER0_OF,
    prsConsumerIADC0_SCANTRIGGER, ldmaCtrlSizeWord
};
static constexpr int ECODE_EMDRV_DMADRV_OK = 0, IADC_IF_SCANFIFOOF = 1;
// Peripheral calls are inert: host tests feed PCM through the real audio ring buffer.
#define TEST_VOID(name) template<class... Args> static void name(Args...) {}
#define TEST_ZERO(name) template<class... Args> static int name(Args...) { return 0; }
TEST_VOID(CMU_ClockEnable)
TEST_VOID(CMU_ClockSelectSet)
TEST_VOID(IADC_reset)
TEST_ZERO(IADC_calcSrcClkPrescale)
TEST_ZERO(IADC_calcAdcClkPrescale)
TEST_VOID(IADC_init)
TEST_VOID(IADC_initScan)
TEST_ZERO(IADC_getInt)
TEST_VOID(IADC_clearInt)
TEST_VOID(IADC_command)
TEST_VOID(DMADRV_Init)
TEST_ZERO(DMADRV_AllocateChannel)
TEST_ZERO(DMADRV_LdmaStartTransfer)
TEST_VOID(TIMER_Init)
TEST_VOID(TIMER_TopSet)
TEST_VOID(TIMER_TopBufSet)
TEST_VOID(TIMER_Enable)
TEST_VOID(PRS_ConnectSignal)
TEST_VOID(PRS_ConnectConsumer)
static uint32_t CMU_ClockFreqGet(int) { return 78000000; }
static constexpr int SL_POWER_MANAGER_EM1 = 1;
TEST_VOID(sl_power_manager_init)
TEST_VOID(sl_power_manager_add_em_requirement)
#undef TEST_VOID
#undef TEST_ZERO
