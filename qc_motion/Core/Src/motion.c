#include "motion.h"
#include "tmc5160.h"
#include <stdio.h>

/*
 * Field mapping (SWT35 XY):
 *   x -> J8 / C
 *   y -> J7 / B
 *   z -> J6 / A   leftover socket
 *
 * Lead 4 mm, 1.8° motor, 32 µstep. Measured 2026-10-12:
 *   a 20 mm command traveled 16 mm at 1280 step/mm, both axes.
 *   200 * 32 / 4 = 1600 step/mm
 *   Origin is the bottom-left corner (left motor + near black housing).
 *   Driver DIR pins point the other way, so X/Y are inverted here.
 *   x+ : toward the green PCB
 *   y+ : toward the far rail with the scale
 */
#define STEPS_PER_MM   1600
#define JOG_MM         2
#define JOG_STEPS      (STEPS_PER_MM * JOG_MM)
#define STEP_GAP_US    80U

static volatile uint8_t abort_jog;
static int32_t pos_steps[2];

static const char axis_name[] = {'X', 'Y', 'Z'};
static const char sock_name[] = {'A', 'B', 'C'};

static uint8_t axis_ch(motion_axis_t axis)
{
    if (axis == AXIS_X) {
        return TMC_CH_C;
    }
    if (axis == AXIS_Y) {
        return TMC_CH_B;
    }
    return TMC_CH_A;
}

static void delay_us(uint32_t us)
{
    uint32_t start = DWT->CYCCNT;
    uint32_t ticks = us * (SystemCoreClock / 1000000UL);
    while ((DWT->CYCCNT - start) < ticks) {
    }
}

static void print_mm_steps(int32_t steps)
{
    int32_t milli;
    const char *sign = "";

    if (steps < 0) {
        sign = "-";
        steps = -steps;
    }
    milli = (int32_t)(((int64_t)steps * 1000LL) / STEPS_PER_MM);
    printf("%s%ld.%03ld", sign, (long)(milli / 1000), (long)(milli % 1000));
}

static void add_pos(motion_axis_t axis, int dir, uint32_t done)
{
    if (axis > AXIS_Y || done == 0U) {
        return;
    }
    if (dir >= 0) {
        pos_steps[axis] += (int32_t)done;
    } else {
        pos_steps[axis] -= (int32_t)done;
    }
}

static int wire_dir(motion_axis_t axis, int dir)
{
    if (axis == AXIS_X || axis == AXIS_Y) {
        return -dir;
    }
    return dir;
}

static uint32_t run_steps(motion_axis_t axis, int dir, uint32_t steps)
{
    uint8_t ch = axis_ch(axis);
    uint32_t done = 0U;

    if (steps == 0U || abort_jog) {
        return 0U;
    }

    tmc5160_enable(ch, 1U);
    delay_us(200U);
    tmc5160_set_dir(ch, wire_dir(axis, dir));
    delay_us(10U);
    while (done < steps) {
        if (abort_jog) {
            break;
        }
        tmc5160_step_pulse(ch);
        delay_us(STEP_GAP_US);
        done++;
    }
    tmc5160_enable(ch, 0U);
    return done;
}

void Motion_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    TMC5160_Init();
    abort_jog = 0U;
    pos_steps[AXIS_X] = 0;
    pos_steps[AXIS_Y] = 0;
}

void Motion_RequestStop(void)
{
    abort_jog = 1U;
}

void Motion_Stop(void)
{
    abort_jog = 1U;
}

void Motion_Zero(void)
{
    pos_steps[AXIS_X] = 0;
    pos_steps[AXIS_Y] = 0;
    printf("zero  x=0.000 y=0.000\r\n");
}

void Motion_PrintPos(void)
{
    printf("pos   x=");
    print_mm_steps(pos_steps[AXIS_X]);
    printf(" y=");
    print_mm_steps(pos_steps[AXIS_Y]);
    printf("\r\n");
}

void Motion_Jog(motion_axis_t axis, int dir)
{
    uint8_t ch = axis_ch(axis);
    uint32_t done;
    char ax = axis_name[(unsigned)axis];
    char sk = sock_name[ch];

    if (!tmc5160_chip_ok(ch)) {
        printf("err  tmc %c not ready\r\n", sk);
        return;
    }

    abort_jog = 0U;
    printf("jog  %c%s  sock %c  2mm\r\n", ax, (dir >= 0) ? "+" : "-", sk);
    done = run_steps(axis, dir, (uint32_t)JOG_STEPS);
    add_pos(axis, dir, done);
    printf("jog  done  steps=%lu\r\n", (unsigned long)done);
}

static int move_axis_to(motion_axis_t axis, int32_t target_steps)
{
    int32_t delta = target_steps - pos_steps[axis];
    int dir = 1;
    uint8_t ch = axis_ch(axis);
    uint32_t done;

    if (delta == 0) {
        return 1;
    }
    if (delta < 0) {
        dir = -1;
        delta = -delta;
    }
    if (!tmc5160_chip_ok(ch)) {
        printf("err  tmc %c not ready\r\n", sock_name[ch]);
        return 0;
    }
    done = run_steps(axis, dir, (uint32_t)delta);
    add_pos(axis, dir, done);
    return (done == (uint32_t)delta) && !abort_jog;
}

void Motion_MoveAbs(int32_t x_milli_mm, int32_t y_milli_mm)
{
    int32_t x_steps;
    int32_t y_steps;
    int ok;

    x_steps = (int32_t)(((int64_t)x_milli_mm * STEPS_PER_MM) / 1000);
    y_steps = (int32_t)(((int64_t)y_milli_mm * STEPS_PER_MM) / 1000);

    abort_jog = 0U;
    printf("move  x=");
    print_mm_steps(x_steps);
    printf(" y=");
    print_mm_steps(y_steps);
    printf("\r\n");

    ok = move_axis_to(AXIS_X, x_steps);
    if (ok) {
        ok = move_axis_to(AXIS_Y, y_steps);
    }
    if (!ok) {
        printf("move  stopped\r\n");
    }
    Motion_PrintPos();
}

void Motion_Poll(void)
{
}
