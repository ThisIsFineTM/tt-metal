MEMORY
{
#if defined(COMPILE_FOR_BRISC)
    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_BRISC_LOCAL_SIZE - MEM_BRISC_STACK_SIZE
    REGION_STACK : ORIGIN = MEM_BRISC_STACK_TOP - MEM_BRISC_STACK_SIZE, LENGTH = MEM_BRISC_STACK_SIZE
#if defined(TYPE_FIRMWARE)
    REGION_CODE : ORIGIN = MEM_BRISC_FIRMWARE_BASE, LENGTH = MEM_BRISC_FIRMWARE_SIZE
#else
    REGION_CODE : ORIGIN = MEM_BRISC_FIRMWARE_BASE, LENGTH = MEM_BRISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_NCRISC)
    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_NCRISC_LOCAL_SIZE - MEM_NCRISC_STACK_SIZE
    REGION_STACK : ORIGIN = MEM_NCRISC_STACK_TOP - MEM_NCRISC_STACK_SIZE, LENGTH = MEM_NCRISC_STACK_SIZE
#if defined(TYPE_FIRMWARE)
    REGION_CODE : ORIGIN = MEM_NCRISC_FIRMWARE_BASE, LENGTH = MEM_NCRISC_FIRMWARE_SIZE
#elif defined(NCRISC_FIRMWARE_KERNEL_SPLIT)
    REGION_CODE : ORIGIN = MEM_NCRISC_KERNEL_BASE, LENGTH = MEM_NCRISC_KERNEL_SIZE
#else
    REGION_CODE : ORIGIN = MEM_NCRISC_FIRMWARE_BASE, LENGTH = MEM_NCRISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_IERISC)
    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_IERISC_LOCAL_SIZE - MEM_IERISC_STACK_SIZE
    REGION_STACK : ORIGIN = MEM_IERISC_STACK_TOP - MEM_IERISC_STACK_SIZE, LENGTH = MEM_IERISC_STACK_SIZE
#if defined(TYPE_FIRMWARE)
    REGION_CODE : ORIGIN = MEM_IERISC_FIRMWARE_BASE, LENGTH = MEM_IERISC_FIRMWARE_SIZE
#else
    REGION_CODE : ORIGIN = MEM_IERISC_FIRMWARE_BASE, LENGTH = MEM_IERISC_KERNEL_SIZE
#endif

#elif defined(COMPILE_FOR_SLAVE_IERISC)
    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_SLAVE_IERISC_LOCAL_SIZE - MEM_SLAVE_IERISC_STACK_SIZE
    REGION_STACK : ORIGIN = MEM_SLAVE_IERISC_STACK_TOP - MEM_SLAVE_IERISC_STACK_SIZE, LENGTH = MEM_SLAVE_IERISC_STACK_SIZE
    REGION_CODE : ORIGIN = MEM_SLAVE_IERISC_FIRMWARE_BASE, LENGTH = MEM_SLAVE_IERISC_FIRMWARE_SIZE

#elif defined(COMPILE_FOR_TRISC)
#define TRISC_SELECT__(BEFORE,MIDDLE,AFTER) BEFORE##TRISC##MIDDLE##AFTER
#define TRISC_SELECT_(BEFORE,MIDDLE,AFTER) TRISC_SELECT__(BEFORE,MIDDLE,AFTER)
#define TRISC_SELECT(BEFORE,AFTER) TRISC_SELECT_(BEFORE,COMPILE_FOR_TRISC,AFTER)

    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_TRISC_LOCAL_SIZE - TRISC_SELECT(MEM_,_STACK_SIZE)
    REGION_STACK : ORIGIN = TRISC_SELECT(MEM_,_STACK_TOP) - TRISC_SELECT(MEM_,_STACK_SIZE), LENGTH = TRISC_SELECT(MEM_,_STACK_SIZE)
#if defined(TYPE_FIRMWARE)
    REGION_CODE : ORIGIN = TRISC_SELECT(MEM_,_FIRMWARE_BASE), LENGTH = TRISC_SELECT(MEM_,_FIRMWARE_SIZE)
#else
    REGION_CODE : ORIGIN = TRISC_SELECT(MEM_,_FIRMWARE_BASE), LENGTH = TRISC_SELECT(MEM_,_KERNEL_SIZE)
#endif

#elif defined(COMPILE_FOR_AERISC)
    REGION_DATA : ORIGIN = MEM_LOCAL_BASE, LENGTH = MEM_IERISC_LOCAL_SIZE - MEM_IERISC_STACK_SIZE
    REGION_STACK : ORIGIN = MEM_AERISC_STACK_TOP - MEM_IERISC_STACK_SIZE, LENGTH = MEM_IERISC_STACK_SIZE
#if defined(TYPE_FIRMWARE)
    REGION_CODE : ORIGIN = MEM_AERISC_FIRMWARE_BASE, LENGTH = MEM_IERISC_FIRMWARE_SIZE
#else
    REGION_CODE : ORIGIN = MEM_AERISC_FIRMWARE_BASE, LENGTH = MEM_IERISC_KERNEL_SIZE
#endif

#else
#error "compiling for unknown"
#endif
}
