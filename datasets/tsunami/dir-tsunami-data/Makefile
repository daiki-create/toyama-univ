
OBJ_DIR = ${HOME}/obj
include ${OBJ_DIR}/OBJ.make

mk_dataset_seismic_tsunami:mk_dataset_seismic_tsunami.F90 \
             ${OBJ_DIR}/prjctn.o        ${OBJ_DIR}/pltak.o \
             ${OBJ_DIR}/timsub.o        ${OBJ_DIR}/calpakk_v.o \
             ${OBJ_DIR}/dgtfilt_d.o     ${OBJ_DIR}/besfilt_d.o  \
             ${OBJ_DIR}/tsdtsc.o        ${OBJ_DIR}/trtpsh4_.o  \
             ${OBJ_DIR}/trtime4_.o      ${OBJ_DIR}/getllst2_.o  \
             ${OBJ_DIR}/dtoabc_.o       ${OBJ_DIR}/delta.o \
             ${OBJ_DIR}/sttab3d_.o      ${OBJ_DIR}/quick_sort_fort.o
	${FC} ${FCFLAGS} -o $@ $< \
	${INCLUDES} \
	${OBJ_DIR}/prjctn.o        ${OBJ_DIR}/pltak.o \
	${OBJ_DIR}/timsub.o        ${OBJ_DIR}/calpakk_v.o \
	${OBJ_DIR}/dgtfilt_d.o     ${OBJ_DIR}/besfilt_d.o \
	${OBJ_DIR}/tsdtsc.o        ${OBJ_DIR}/trtpsh4_.o  \
	${OBJ_DIR}/trtime4_.o      ${OBJ_DIR}/getllst2_.o  \
	${OBJ_DIR}/dtoabc_.o       ${OBJ_DIR}/delta.o \
	${OBJ_DIR}/sttab3d_.o      ${OBJ_DIR}/quick_sort_fort.o \
	-lm -lc

