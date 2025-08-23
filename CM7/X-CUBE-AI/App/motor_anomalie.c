/**
  ******************************************************************************
  * @file    motor_anomalie.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-08-23T15:13:00+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "motor_anomalie.h"
#include "motor_anomalie_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_motor_anomalie
 
#undef AI_MOTOR_ANOMALIE_MODEL_SIGNATURE
#define AI_MOTOR_ANOMALIE_MODEL_SIGNATURE     "0xffa8546049c20e4326a03650d45f6a27"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-08-23T15:13:00+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_MOTOR_ANOMALIE_N_BATCHES
#define AI_MOTOR_ANOMALIE_N_BATCHES         (1)

static ai_ptr g_motor_anomalie_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_motor_anomalie_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_keras_tensor0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 180, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  pool_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 960, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  pool_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 960, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1216, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 960, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  pool_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 4, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 864, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14336, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20480, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2284, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6912, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7296, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 384, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 84, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 124, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.059185463935136795f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017358871176838875f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009505628840997815f, 0.0009323331760242581f, 0.0010279176058247685f, 0.001001707511022687f, 0.0007992373430170119f, 0.0009229978895746171f, 0.0009611013811081648f, 0.0009668418206274509f, 0.0009683345560915768f, 0.0010338635183870792f, 0.0009104391792789102f, 0.0008942974964156747f, 0.0010537686757743359f, 0.0008739611366763711f, 0.0009542278130538762f, 0.0009936850983649492f, 0.0009671316947788f, 0.000893728865776211f, 0.0008915510261431336f, 0.0008497886010445654f, 0.0009256022167392075f, 0.0009765976574271917f, 0.000956302392296493f, 0.0007985108532011509f, 0.0010449446272104979f, 0.000856749014928937f, 0.0009756999788805842f, 0.0009643326047807932f, 0.0009231188450939953f, 0.0009739266824908555f, 0.0009769260650500655f, 0.0010195462964475155f, 0.0009157808963209391f, 0.0010107529815286398f, 0.0008311629062518477f, 0.0008225206402130425f, 0.0009434031089767814f, 0.001010590698570013f, 0.0008104043663479388f, 0.0010652923956513405f, 0.0009923770558089018f, 0.0010712011717259884f, 0.0008412766037508845f, 0.001036162138916552f, 0.0010206683073192835f, 0.0009756405488587916f, 0.001078415778465569f, 0.0007599751115776598f, 0.000932676310185343f, 0.001101282425224781f, 0.0009839803678914905f, 0.0009298350778408349f, 0.0009856177493929863f, 0.0010555877815932035f, 0.0010170433670282364f, 0.000940776604693383f, 0.0008290203986689448f, 0.000978337600827217f, 0.0009832537034526467f, 0.0010304353199899197f, 0.0010150811867788434f, 0.0009743249393068254f, 0.000920823949854821f, 0.0008377534104511142f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010333019308745861f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013153563486412168f, 0.0013348069041967392f, 0.0012169339461252093f, 0.0012233221204951406f, 0.0013303278246894479f, 0.0011701049515977502f, 0.0013354087714105844f, 0.0012688605347648263f, 0.001239053439348936f, 0.0012651507277041674f, 0.0012387369060888886f, 0.0012080010492354631f, 0.001211959170177579f, 0.001258305972442031f, 0.0011221586028113961f, 0.0012687906855717301f, 0.0011613089591264725f, 0.0013152153696864843f, 0.0013172869803383946f, 0.0012391223572194576f, 0.0012228790437802672f, 0.0012681831140071154f, 0.0011912541231140494f, 0.0011861390667036176f, 0.0013181633548811078f, 0.0012794028734788299f, 0.0012706337729468942f, 0.001317173126153648f, 0.0011330742854624987f, 0.0012175280135124922f, 0.001152894925326109f, 0.0012794769136235118f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017358871176838875f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010333019308745861f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008021871326491237f, 0.0009499246953055263f, 0.001047243713401258f, 0.0009258738718926907f, 0.000902288593351841f, 0.0008455583592876792f, 0.0009393775835633278f, 0.0008251420222222805f, 0.0009196982719004154f, 0.0008821391384117305f, 0.0010667325695976615f, 0.000846049573738128f, 0.0009448426426388323f, 0.000925293774344027f, 0.0008956188685260713f, 0.0009496745187789202f, 0.0009365109144710004f, 0.0009259582730010152f, 0.0008038728847168386f, 0.0009334452915936708f, 0.0009200244094245136f, 0.0009342497796751559f, 0.0009507699869573116f, 0.0009622117504477501f, 0.000837744795717299f, 0.0009110355167649686f, 0.0009284177212975919f, 0.0009082347387447953f, 0.0009375280351378024f, 0.0008760397904552519f, 0.0009366216254420578f, 0.0009945277124643326f, 0.0009036968694999814f, 0.0009498186991550028f, 0.0009575174772180617f, 0.0008723067003302276f, 0.0010975634213536978f, 0.0008882961701601744f, 0.00091377372154966f, 0.0009572568233124912f, 0.0008311433484777808f, 0.0009717295761220157f, 0.0008928254246711731f, 0.0009270081645809114f, 0.0008336401078850031f, 0.0009669331484474242f, 0.0009177952306345105f, 0.0008358114282600582f, 0.0007658345275558531f, 0.0008585898904129863f, 0.000985106104053557f, 0.0009836471872404218f, 0.0009256845805794001f, 0.0009695373591966927f, 0.0009216057951562107f, 0.000873675977345556f, 0.0009144864743575454f, 0.0009454709361307323f, 0.0009086521458812058f, 0.0009194804006256163f, 0.0009297490469180048f, 0.0008934178040362895f, 0.0009318623342551291f, 0.0008505774312652647f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0547766387462616f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017525668954476714f, 0.001705359434708953f, 0.0019672547932714224f, 0.0017354326555505395f, 0.001821568701416254f, 0.0018392518395558f, 0.0016759592108428478f, 0.0019894586876034737f, 0.001812976785004139f, 0.0019373362883925438f, 0.0016675591468811035f, 0.001785410800948739f, 0.0018546448554843664f, 0.001639611436985433f, 0.0016922814538702369f, 0.0019248033640906215f, 0.0019666901789605618f, 0.0018370688194409013f, 0.0018522415775805712f, 0.0019713223446160555f, 0.0017401494551450014f, 0.001832790207117796f, 0.0019166937563568354f, 0.0017263522604480386f, 0.001770580536685884f, 0.0019388385117053986f, 0.0018636920722201467f, 0.0016964190872386098f, 0.0017411791486665606f, 0.0016698440304026008f, 0.0017966779414564371f, 0.0016225975705310702f, 0.001836359966546297f, 0.0019203368574380875f, 0.0019453573040664196f, 0.00189626170322299f, 0.0017098687821999192f, 0.0019873592536896467f, 0.0016895317239686847f, 0.001742977648973465f, 0.0017982295248657465f, 0.0018583027413114905f, 0.0018605329096317291f, 0.0017046143766492605f, 0.0017393733141943812f, 0.0018433372024446726f, 0.0017503605922684073f, 0.00191090430598706f, 0.0018512799870222807f, 0.001823639264330268f, 0.0017925275024026632f, 0.0019321549916639924f, 0.0019948813132941723f, 0.0018469460774213076f, 0.0016802713507786393f, 0.0019158717477694154f, 0.0017249900847673416f, 0.001743230503052473f, 0.0016184430569410324f, 0.0017390786670148373f, 0.0018571646651253104f, 0.0017331173876300454f, 0.001794050564058125f, 0.0017683333717286587f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1428515762090683f),
    AI_PACK_INTQ_ZP(34)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002557568484917283f, 0.002591637661680579f, 0.0025063608773052692f, 0.002569126430898905f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017358871176838875f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04002843797206879f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010333019308745861f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_keras_tensor0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025338666513562202f),
    AI_PACK_INTQ_ZP(12)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_13_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 15, 1), AI_STRIDE_INIT(4, 1, 1, 64, 960),
  1, &conv2d_13_output_array, &conv2d_13_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output0, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 15), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_13_output_array, &conv2d_13_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_pad_before_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 19, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1216),
  1, &conv2d_13_pad_before_output_array, &conv2d_13_pad_before_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 7296, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7296, 7296),
  1, &conv2d_13_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_weights, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 64, 5, 1, 64), AI_STRIDE_INIT(4, 1, 64, 4096, 20480),
  1, &conv2d_13_weights_array, &conv2d_13_weights_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 60, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1920),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 2284, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2284, 2284),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 3, 9, 1, 32), AI_STRIDE_INIT(4, 1, 3, 96, 864),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 30, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1920),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_pad_before_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_7_pad_before_output_array, &conv2d_7_pad_before_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 6912, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6912, 6912),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 32, 7, 1, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 14336),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_16_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_16_output_array, &gemm_16_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 384, 1, 1), AI_STRIDE_INIT(4, 2, 2, 768, 768),
  1, &gemm_16_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_weights, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 64, 4096, 4096),
  1, &gemm_16_weights_array, &gemm_16_weights_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &gemm_17_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4, 4),
  1, &gemm_17_output_array, &gemm_17_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 84, 1, 1), AI_STRIDE_INIT(4, 2, 2, 168, 168),
  1, &gemm_17_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_weights, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 64, 4, 1, 1), AI_STRIDE_INIT(4, 1, 64, 256, 256),
  1, &gemm_17_weights_array, &gemm_17_weights_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4, 4),
  1, &nl_18_output_array, &nl_18_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 124, 1, 1), AI_STRIDE_INIT(4, 4, 4, 496, 496),
  1, &nl_18_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 15, 1), AI_STRIDE_INIT(4, 1, 1, 64, 960),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  pool_15_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_15_output_array, &pool_15_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  pool_4_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 30, 1), AI_STRIDE_INIT(4, 1, 1, 32, 960),
  1, &pool_4_output_array, &pool_4_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor0_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 1, 60), AI_STRIDE_INIT(4, 1, 1, 3, 3),
  1, &serving_default_keras_tensor0_output_array, &serving_default_keras_tensor0_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor0_output0, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 60, 1), AI_STRIDE_INIT(4, 1, 1, 3, 180),
  1, &serving_default_keras_tensor0_output_array, &serving_default_keras_tensor0_output_array_intq)



/**  Layer declarations section  **********************************************/



AI_STATIC_CONST ai_i32 nl_18_nl_params_data[] = { 1227085696, 24, -124 };
AI_ARRAY_OBJ_DECLARE(
    nl_18_nl_params, AI_ARRAY_FORMAT_S32,
    nl_18_nl_params_data, nl_18_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_18_layer, 18,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_18_chain,
  NULL, &nl_18_layer, AI_STATIC, 
  .nl_params = &nl_18_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_17_weights, &gemm_17_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_17_layer, 17,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_17_chain,
  NULL, &nl_18_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_16_weights, &gemm_16_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_16_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_16_chain,
  NULL, &gemm_17_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_15_layer, 15,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_15_chain,
  NULL, &gemm_16_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 15), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 15), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_13_weights, &conv2d_13_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_13_chain,
  NULL, &pool_15_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_13_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_13_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_13_pad_before_value_data, conv2d_13_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_pad_before_layer, 13,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_13_pad_before_chain,
  NULL, &conv2d_13_layer, AI_STATIC, 
  .value = &conv2d_13_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 2, 0, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_10_layer, 10,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_10_chain,
  NULL, &conv2d_13_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_7_chain,
  NULL, &pool_10_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 conv2d_7_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_7_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_7_pad_before_value_data, conv2d_7_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_pad_before_layer, 7,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_7_pad_before_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .value = &conv2d_7_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 3, 0, 3), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_4_layer, 4,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_4_chain,
  NULL, &conv2d_7_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_keras_tensor0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_1_chain,
  NULL, &pool_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 4, 0, 4), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 40944, 1, 1),
    40944, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 9984, 1, 1),
    9984, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MOTOR_ANOMALIE_IN_NUM, &serving_default_keras_tensor0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MOTOR_ANOMALIE_OUT_NUM, &nl_18_output),
  &conv2d_1_layer, 0x100f3956, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 40944, 1, 1),
      40944, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 9984, 1, 1),
      9984, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MOTOR_ANOMALIE_IN_NUM, &serving_default_keras_tensor0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MOTOR_ANOMALIE_OUT_NUM, &nl_18_output),
  &conv2d_1_layer, 0x100f3956, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool motor_anomalie_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_motor_anomalie_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_keras_tensor0_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 960);
    serving_default_keras_tensor0_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 960);
    conv2d_1_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 1140);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 1140);
    conv2d_1_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 3424);
    conv2d_1_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 3424);
    pool_4_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 960);
    pool_4_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 960);
    conv2d_7_pad_before_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 1920);
    conv2d_7_pad_before_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 1920);
    conv2d_7_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 3072);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 3072);
    conv2d_7_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    conv2d_7_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    pool_10_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 1920);
    pool_10_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 1920);
    conv2d_13_pad_before_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    conv2d_13_pad_before_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    conv2d_13_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 1216);
    conv2d_13_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 1216);
    conv2d_13_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 8512);
    conv2d_13_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 8512);
    pool_15_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    pool_15_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    gemm_16_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 64);
    gemm_16_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 64);
    gemm_16_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 832);
    gemm_16_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 832);
    gemm_17_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    gemm_17_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    gemm_17_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 168);
    gemm_17_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 168);
    nl_18_scratch0_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 172);
    nl_18_scratch0_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 172);
    nl_18_output_array.data = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    nl_18_output_array.data_start = AI_PTR(g_motor_anomalie_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool motor_anomalie_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_motor_anomalie_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 0);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 864);
    conv2d_1_bias_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 864);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 992);
    conv2d_7_weights_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 992);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 15328);
    conv2d_7_bias_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 15328);
    conv2d_13_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_weights_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 15584);
    conv2d_13_weights_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 15584);
    conv2d_13_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_bias_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 36064);
    conv2d_13_bias_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 36064);
    gemm_16_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_weights_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 36320);
    gemm_16_weights_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 36320);
    gemm_16_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_bias_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 40416);
    gemm_16_bias_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 40416);
    gemm_17_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_17_weights_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 40672);
    gemm_17_weights_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 40672);
    gemm_17_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_17_bias_array.data = AI_PTR(g_motor_anomalie_weights_map[0] + 40928);
    gemm_17_bias_array.data_start = AI_PTR(g_motor_anomalie_weights_map[0] + 40928);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_motor_anomalie_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MOTOR_ANOMALIE_MODEL_NAME,
      .model_signature   = AI_MOTOR_ANOMALIE_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 798560,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x100f3956,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_motor_anomalie_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MOTOR_ANOMALIE_MODEL_NAME,
      .model_signature   = AI_MOTOR_ANOMALIE_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 798560,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x100f3956,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_motor_anomalie_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_motor_anomalie_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_motor_anomalie_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_motor_anomalie_create(network, AI_MOTOR_ANOMALIE_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_motor_anomalie_data_params_get(&params) != true) {
    err = ai_motor_anomalie_get_error(*network);
    return err;
  }
#if defined(AI_MOTOR_ANOMALIE_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_MOTOR_ANOMALIE_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_motor_anomalie_init(*network, &params) != true) {
    err = ai_motor_anomalie_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_motor_anomalie_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_motor_anomalie_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_motor_anomalie_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_motor_anomalie_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= motor_anomalie_configure_weights(net_ctx, params);
  ok &= motor_anomalie_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_motor_anomalie_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_motor_anomalie_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_MOTOR_ANOMALIE_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

