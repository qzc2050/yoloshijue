#include "tmc5160.h"
#include <stdio.h>

/* CS: A=PB6  B=PB7  C=PB2 */
#define SM_CS1_Pin          GPIO_PIN_6
#define SM_CS1_GPIO_Port    GPIOB
#define SM_CS2_Pin          GPIO_PIN_7
#define SM_CS2_GPIO_Port    GPIOB
#define SM_CS3_Pin          GPIO_PIN_2
#define SM_CS3_GPIO_Port    GPIOB

#define SM_SCK_Pin          GPIO_PIN_3
#define SM_MOSI_Pin         GPIO_PIN_5
#define SM_MISO_Pin         GPIO_PIN_4

/* EN active-low DRV_ENN: A=PE13  B=PE0  C=PD4  (schematic ADRV_EN/BDRV_EN/CDRV_EN) */
#define SM_EN1_Pin          GPIO_PIN_13
#define SM_EN1_GPIO_Port    GPIOE
#define SM_EN2_Pin          GPIO_PIN_0
#define SM_EN2_GPIO_Port    GPIOE
#define SM_EN3_Pin          GPIO_PIN_4
#define SM_EN3_GPIO_Port    GPIOD

/* This board runs SPI+STEP/DIR (AS600 CONTROL_MODE=1). SD_MODE must be 1. */
#define SM_ASD_MODE_Pin     GPIO_PIN_11
#define SM_ASD_MODE_Port    GPIOE
#define SM_BSD_MODE_Pin     GPIO_PIN_10
#define SM_BSD_MODE_Port    GPIOB
#define SM_CSD_MODE_Pin     GPIO_PIN_2
#define SM_CSD_MODE_Port    GPIOD

#define SM_ASPI_MODE_Pin    GPIO_PIN_12
#define SM_ASPI_MODE_Port   GPIOE
#define SM_BSPI_MODE_Pin    GPIO_PIN_11
#define SM_BSPI_MODE_Port   GPIOB
#define SM_CSPI_MODE_Pin    GPIO_PIN_3
#define SM_CSPI_MODE_Port   GPIOD

#define TMC5160_MRES_SHIFT  24U
#define TMC_CHOPCONF_ON     (0x00410153U | (3UL << TMC5160_MRES_SHIFT))
#define TMC_CHOPCONF_OFF    (TMC_CHOPCONF_ON & ~0x0FUL) /* TOFF=0: bridges off */

/* A=X PD13/PE14  B=Y PD14/PE1  C=Z PD15/PD5 */
#define STEP_A_Pin          GPIO_PIN_13
#define STEP_A_Port         GPIOD
#define DIR_A_Pin           GPIO_PIN_14
#define DIR_A_Port          GPIOE
#define STEP_B_Pin          GPIO_PIN_14
#define STEP_B_Port         GPIOD
#define DIR_B_Pin           GPIO_PIN_1
#define DIR_B_Port          GPIOE
#define STEP_C_Pin          GPIO_PIN_15
#define STEP_C_Port         GPIOD
#define DIR_C_Pin           GPIO_PIN_5
#define DIR_C_Port          GPIOD

static SPI_HandleTypeDef hspi1;
static uint8_t chip_ok[3];

static void delay_us(uint32_t us)
{
    uint32_t cycles = (SystemCoreClock / 1000000UL) * us;
    volatile uint32_t i = cycles / 4U;
    while (i--) {
        __NOP();
    }
}

static void tmc5160_select(uint8_t ch, uint8_t high)
{
    GPIO_PinState st = high ? GPIO_PIN_SET : GPIO_PIN_RESET;
    if (ch == 0U) {
        HAL_GPIO_WritePin(SM_CS1_GPIO_Port, SM_CS1_Pin, st);
    } else if (ch == 1U) {
        HAL_GPIO_WritePin(SM_CS2_GPIO_Port, SM_CS2_Pin, st);
    } else {
        HAL_GPIO_WritePin(SM_CS3_GPIO_Port, SM_CS3_Pin, st);
    }
    delay_us(1U);
}

static uint8_t spi_xfer(uint8_t tx)
{
    uint8_t rx = 0U;
    (void)HAL_SPI_TransmitReceive(&hspi1, &tx, &rx, 1U, 20U);
    return rx;
}

void tmc5160_write(uint8_t ch, uint8_t reg, uint32_t data)
{
    tmc5160_select(ch, 0U);
    (void)spi_xfer((uint8_t)(reg | 0x80U));
    (void)spi_xfer((uint8_t)((data >> 24) & 0xFFU));
    (void)spi_xfer((uint8_t)((data >> 16) & 0xFFU));
    (void)spi_xfer((uint8_t)((data >> 8) & 0xFFU));
    (void)spi_xfer((uint8_t)(data & 0xFFU));
    tmc5160_select(ch, 1U);
}

uint32_t tmc5160_read(uint8_t ch, uint8_t reg)
{
    uint32_t value;
    uint8_t i;

    for (i = 0U; i < 2U; i++) {
        value = 0U;
        tmc5160_select(ch, 0U);
        (void)spi_xfer((uint8_t)(reg & 0x7FU));
        value |= ((uint32_t)spi_xfer(0U)) << 24;
        value |= ((uint32_t)spi_xfer(0U)) << 16;
        value |= ((uint32_t)spi_xfer(0U)) << 8;
        value |= (uint32_t)spi_xfer(0U);
        tmc5160_select(ch, 1U);
    }
    return value;
}

uint8_t tmc5160_chip_ok(uint8_t ch)
{
    return (ch < 3U) ? chip_ok[ch] : 0U;
}

void tmc5160_set_dir(uint8_t ch, int dir)
{
    GPIO_PinState st = (dir >= 0) ? GPIO_PIN_SET : GPIO_PIN_RESET;

    if (ch == TMC_CH_A) {
        HAL_GPIO_WritePin(DIR_A_Port, DIR_A_Pin, st);
    } else if (ch == TMC_CH_B) {
        HAL_GPIO_WritePin(DIR_B_Port, DIR_B_Pin, st);
    } else if (ch == TMC_CH_C) {
        HAL_GPIO_WritePin(DIR_C_Port, DIR_C_Pin, st);
    }
}

void tmc5160_step_pulse(uint8_t ch)
{
    if (ch == TMC_CH_A) {
        HAL_GPIO_WritePin(STEP_A_Port, STEP_A_Pin, GPIO_PIN_SET);
        delay_us(2U);
        HAL_GPIO_WritePin(STEP_A_Port, STEP_A_Pin, GPIO_PIN_RESET);
    } else if (ch == TMC_CH_B) {
        HAL_GPIO_WritePin(STEP_B_Port, STEP_B_Pin, GPIO_PIN_SET);
        delay_us(2U);
        HAL_GPIO_WritePin(STEP_B_Port, STEP_B_Pin, GPIO_PIN_RESET);
    } else if (ch == TMC_CH_C) {
        HAL_GPIO_WritePin(STEP_C_Port, STEP_C_Pin, GPIO_PIN_SET);
        delay_us(2U);
        HAL_GPIO_WritePin(STEP_C_Port, STEP_C_Pin, GPIO_PIN_RESET);
    }
}

static void tmc_gpio_init(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();

    HAL_GPIO_WritePin(SM_CS1_GPIO_Port, SM_CS1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_CS2_GPIO_Port, SM_CS2_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_CS3_GPIO_Port, SM_CS3_Pin, GPIO_PIN_SET);

    /* disable all drivers until chips answer */
    HAL_GPIO_WritePin(SM_EN1_GPIO_Port, SM_EN1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_EN2_GPIO_Port, SM_EN2_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_EN3_GPIO_Port, SM_EN3_Pin, GPIO_PIN_SET);

    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_PULLUP;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    gpio.Pin = SM_CS1_Pin | SM_CS2_Pin | SM_CS3_Pin;
    HAL_GPIO_Init(GPIOB, &gpio);

    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;

    gpio.Pin = SM_EN1_Pin;
    HAL_GPIO_Init(SM_EN1_GPIO_Port, &gpio);
    gpio.Pin = SM_EN2_Pin;
    HAL_GPIO_Init(SM_EN2_GPIO_Port, &gpio);
    gpio.Pin = SM_EN3_Pin;
    HAL_GPIO_Init(SM_EN3_GPIO_Port, &gpio);

    gpio.Pin = SM_ASD_MODE_Pin;
    HAL_GPIO_Init(SM_ASD_MODE_Port, &gpio);
    gpio.Pin = SM_BSD_MODE_Pin;
    HAL_GPIO_Init(SM_BSD_MODE_Port, &gpio);
    gpio.Pin = SM_CSD_MODE_Pin;
    HAL_GPIO_Init(SM_CSD_MODE_Port, &gpio);

    gpio.Pin = SM_ASPI_MODE_Pin;
    HAL_GPIO_Init(SM_ASPI_MODE_Port, &gpio);
    gpio.Pin = SM_BSPI_MODE_Pin;
    HAL_GPIO_Init(SM_BSPI_MODE_Port, &gpio);
    gpio.Pin = SM_CSPI_MODE_Pin;
    HAL_GPIO_Init(SM_CSPI_MODE_Port, &gpio);

    gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Pin = STEP_A_Pin | STEP_B_Pin | STEP_C_Pin | DIR_C_Pin;
    HAL_GPIO_Init(GPIOD, &gpio);
    gpio.Pin = DIR_A_Pin | DIR_B_Pin;
    HAL_GPIO_Init(GPIOE, &gpio);
    HAL_GPIO_WritePin(STEP_A_Port, STEP_A_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(STEP_B_Port, STEP_B_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(STEP_C_Port, STEP_C_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(DIR_A_Port, DIR_A_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(DIR_B_Port, DIR_B_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(DIR_C_Port, DIR_C_Pin, GPIO_PIN_RESET);

    /* SPI config + STEP/DIR motion, same as AS600 */
    HAL_GPIO_WritePin(SM_ASPI_MODE_Port, SM_ASPI_MODE_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_BSPI_MODE_Port, SM_BSPI_MODE_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_CSPI_MODE_Port, SM_CSPI_MODE_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_ASD_MODE_Port, SM_ASD_MODE_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_BSD_MODE_Port, SM_BSD_MODE_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SM_CSD_MODE_Port, SM_CSD_MODE_Pin, GPIO_PIN_SET);
}

void HAL_SPI_MspInit(SPI_HandleTypeDef *spi)
{
    GPIO_InitTypeDef gpio = {0};
    RCC_PeriphCLKInitTypeDef clk = {0};

    if (spi->Instance != SPI1) {
        return;
    }

    clk.PeriphClockSelection = RCC_PERIPHCLK_SPI1;
    clk.Spi123ClockSelection = RCC_SPI123CLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&clk) != HAL_OK) {
        Error_Handler();
    }

    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    gpio.Pin = SM_SCK_Pin | SM_MISO_Pin | SM_MOSI_Pin;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    gpio.Alternate = GPIO_AF5_SPI1;
    HAL_GPIO_Init(GPIOB, &gpio);
}

static void tmc_spi_init(void)
{
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_MASTER;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_HIGH;
    hspi1.Init.CLKPhase = SPI_PHASE_2EDGE;
    hspi1.Init.NSS = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    hspi1.Init.CRCPolynomial = 0x0;
    hspi1.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
    hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
    hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
    hspi1.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi1.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
    hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
    hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
    hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_ENABLE;
    hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
    if (HAL_SPI_Init(&hspi1) != HAL_OK) {
        Error_Handler();
    }
}

static void tmc_chip_setup(uint8_t ch)
{
    uint8_t version;

    (void)tmc5160_read(ch, TMC5160_INP_OUT);
    version = (uint8_t)(tmc5160_read(ch, TMC5160_INP_OUT) >> 24);
    chip_ok[ch] = (version == 0x30U) ? 1U : 0U;
    printf("tmc ch%u version=0x%02X %s\r\n",
           (unsigned)ch, (unsigned)version, chip_ok[ch] ? "ok" : "FAIL");
    if (!chip_ok[ch]) {
        return;
    }

    tmc5160_write(ch, TMC5160_GSTAT, 0x07U);
    tmc5160_write(ch, TMC5160_GCONF, 0x00003000U);
    /* SWT35 is 1.0 A. 3 A range: IRUN=10 ≈ 0.94 A, IHOLD=3 idle */
    tmc5160_write(ch, TMC5160_GLOBAL_SCALER, 0U);
    tmc5160_write(ch, TMC5160_IHOLD_IRUN, 0x00070A03U);
    tmc5160_write(ch, TMC5160_TPOWERDOWN, 10U);
    tmc5160_write(ch, TMC5160_CHOPCONF, TMC_CHOPCONF_OFF);
    tmc5160_write(ch, TMC5160_TPWMTHRS, 0U);
    tmc5160_write(ch, TMC5160_SWMODE, 0U);
    printf("tmc ch%u ioin=0x%08lX sd=%lu enn=%lu\r\n",
           (unsigned)ch,
           (unsigned long)tmc5160_read(ch, TMC5160_INP_OUT),
           (unsigned long)((tmc5160_read(ch, TMC5160_INP_OUT) >> 6) & 1UL),
           (unsigned long)((tmc5160_read(ch, TMC5160_INP_OUT) >> 4) & 1UL));
}

void TMC5160_Init(void)
{
    tmc_gpio_init();
    tmc_spi_init();
    HAL_Delay(10);

    tmc_chip_setup(TMC_CH_A);
    tmc_chip_setup(TMC_CH_B);
    tmc_chip_setup(TMC_CH_C);
    HAL_Delay(10);
}

void tmc5160_enable(uint8_t ch, uint8_t on)
{
    GPIO_PinState st = on ? GPIO_PIN_RESET : GPIO_PIN_SET;

    if (ch == TMC_CH_A) {
        HAL_GPIO_WritePin(SM_EN1_GPIO_Port, SM_EN1_Pin, st);
    } else if (ch == TMC_CH_B) {
        HAL_GPIO_WritePin(SM_EN2_GPIO_Port, SM_EN2_Pin, st);
    } else if (ch == TMC_CH_C) {
        HAL_GPIO_WritePin(SM_EN3_GPIO_Port, SM_EN3_Pin, st);
    }
    tmc5160_write(ch, TMC5160_CHOPCONF, on ? TMC_CHOPCONF_ON : TMC_CHOPCONF_OFF);
}
