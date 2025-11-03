//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJXA Image Acquisition Library Header
*/


#ifndef _LJXA_ACQ_H
#define _LJXA_ACQ_H

typedef struct {
	int 	y_linenum;					// Number of profile lines to be acquired in each image.
	int		interpolateLines;			// Number of Interpolate lines.
	float	y_pitch_um;					// Data pitch of Y data. (e.g. your encoder setting)
	int		timeout_ms;					// Timeout error occurs if the acquiring process exceeds the set value.
	int		use_external_batchStart;	// Set "1" if you controll the batch start timing externally. (e.g. terminal input)
} LJXA_ACQ_SETPARAM;

typedef struct {
	int		luminance_enabled;			// Luminance data presence, with luminance: 1, without luminance: 0.
	int		x_pointnum;					// Number of X direction points.
	int		y_linenum_acquired;			// Number of profile lines acquired in each image.
	float	x_pitch_um;					// Data pitch of X data.
	float	y_pitch_um;					// Data pitch of Y data.
	float	z_pitch_um;					// Data pitch of Z data.
} LJXA_ACQ_GETPARAM;

extern "C"
{
int LJXA_ACQ_OpenDevice(int lDeviceId, LJX8IF_ETHERNET_CONFIG *EthernetConfig, int HighSpeedPortNo);

void LJXA_ACQ_CloseDevice(int lDeviceId);

//Blocking I/F
int LJXA_ACQ_Acquire(int lDeviceId, unsigned short *heightImage, unsigned char *luminanceImage, LJXA_ACQ_SETPARAM *setParam, LJXA_ACQ_GETPARAM *getParam);

//Non-blocking I/F
int LJXA_ACQ_StartAsync(int lDeviceId, LJXA_ACQ_SETPARAM *setParam);
int LJXA_ACQ_AcquireAsync(int lDeviceId, unsigned short *heightImage, unsigned char *luminanceImage, LJXA_ACQ_SETPARAM *setParam, LJXA_ACQ_GETPARAM *getParam);

}
#endif /* _LJXA_ACQ_H */