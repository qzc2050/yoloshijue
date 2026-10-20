#ifndef MOTION_H
#define MOTION_H

#include "main.h"

typedef enum {
    AXIS_X = 0,
    AXIS_Y = 1,
    AXIS_Z = 2
} motion_axis_t;

void Motion_Init(void);
void Motion_Jog(motion_axis_t axis, int dir);
void Motion_Zero(void);
void Motion_PrintPos(void);
/* Absolute move. Coordinates are millimeters times 1000, from the last zero. */
void Motion_MoveAbs(int32_t x_milli_mm, int32_t y_milli_mm);
void Motion_Stop(void);
void Motion_RequestStop(void);
void Motion_Poll(void);

#endif
