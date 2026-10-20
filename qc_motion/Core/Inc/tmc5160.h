#ifndef TMC5160_H
#define TMC5160_H

#include "main.h"

/* Board sockets: A=ch0, B=ch1, C=ch2 (same pins as AS600 TMC layer) */
#define TMC_CH_A  0U
#define TMC_CH_B  1U
#define TMC_CH_C  2U

#define TMC5160_GCONF          0x00U
#define TMC5160_GSTAT          0x01U
#define TMC5160_INP_OUT        0x04U
#define TMC5160_DRV_CONF       0x0AU
#define TMC5160_GLOBAL_SCALER  0x0BU
#define TMC5160_IHOLD_IRUN     0x10U
#define TMC5160_TPOWERDOWN     0x11U
#define TMC5160_TPWMTHRS       0x13U
#define TMC5160_RAMPMODE       0x20U
#define TMC5160_XACTUAL        0x21U
#define TMC5160_AMAX           0x26U
#define TMC5160_VMAX           0x27U
#define TMC5160_DMAX           0x28U
#define TMC5160_XTARGET        0x2DU
#define TMC5160_SWMODE         0x34U
#define TMC5160_CHOPCONF       0x6CU

#define TMC5160_MODE_POSITION  0U
#define TMC5160_MODE_VELPOS    1U
#define TMC5160_MODE_VELNEG    2U
#define TMC5160_MODE_HOLD      3U

void TMC5160_Init(void);
void tmc5160_write(uint8_t ch, uint8_t reg, uint32_t data);
uint32_t tmc5160_read(uint8_t ch, uint8_t reg);
uint8_t tmc5160_chip_ok(uint8_t ch);
void tmc5160_enable(uint8_t ch, uint8_t on);
void tmc5160_set_dir(uint8_t ch, int dir);
void tmc5160_step_pulse(uint8_t ch);

#endif
