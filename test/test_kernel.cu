#include <cuda_runtime.h>

#include "device/kernel.cuh"
#include "device/pbkdf2.cuh"
#include "device/romix.cuh"
#include "gtest/gtest.h"
#include "spacemesh_cuda/spacemesh.h"
#include "utils.hpp"

static bool operator==(const uint4 &u0, const uint4 &u1) {
  return u0.x == u1.x && u0.y == u1.y && u0.z == u1.z && u0.w == u1.w;
}

static __global__ void pbkdf2_128b(const uint N, const ulong starting_index,
                                   const uint4 *const __restrict__ input,
                                   uint4 *const __restrict__ output,
                                   const uint num_tasks) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input[0];
    password[1] = input[1];
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();
    /* 1: X = PBKDF2(password, salt) */
    scrypt_pbkdf2_128B(password, X);
    for (uint32_t i = 0; i < 8; ++i) {
      output[t * 8 + i] = X[i];
    }
  }
}

TEST(PBKDF2_128B, CheckResult) {
  const uint32_t N = 8192;
  const uint32_t num_tasks = 8704 * 2;
  const uint64_t starting_index = 0UL;
  CudaDeviceMem<uint4> input(2);
  CudaDeviceMem<uint4> output(8 * num_tasks);
  uint4 h_input[2]{2839345266U, 42009750U,   875455879U,  2217211394U,
                   3438177526U, 2734532412U, 2819254414U, 1408356118U};
  // clang-format off
  // {x[0]xyzw, x[7].xyzw}
  std::vector<std::pair<uint4,uint4>> h_output_ref{
    {{2243168157, 324902921, 784369288, 4178555589}, {594727500, 3520078779, 3153430745, 1486369834}},
    {{2223702091, 3135234577, 351746947, 628596597}, {1974692412, 2751762247, 1815359819, 1220784090}},
    {{1757856292, 3796090370, 105343294, 740218899}, {862746119, 1267304388, 4212448263, 3102108417}},
    {{17321531, 3055860495, 2259029015, 3918725981}, {3736254989, 3761189418, 149153817, 3819126153}},
    {{1834041421, 3427652492, 3278849906, 3382042170}, {3393761384, 947759528, 1750308469, 1815762229}},
    {{3313951662, 1644567330, 866636170, 1422164638}, {1774972657, 1616289065, 2116049991, 2906510373}},
    {{3199039070, 694981869, 1336937698, 1163541043}, {806088862, 1536940888, 2821292057, 915496211}},
    {{4078128437, 2379231243, 1604075742, 2325245807}, {360554255, 875207183, 1516458558, 421131869}},
    {{1087253935, 2208644287, 2756603925, 3971895705}, {1895637869, 3354544041, 2252449461, 218427034}},
    {{1967339238, 1146502695, 3362372873, 2541765279}, {3706574425, 429585357, 9437500, 679403288}},
    {{3457721097, 2061161947, 4289243029, 2170079478}, {2636111503, 3688484586, 4694583, 120348073}},
    {{2852852322, 159818758, 310172246, 2704348751}, {233655115, 1516867167, 2442547836, 936759168}},
    {{3722947891, 3394662328, 1171702661, 3885525270}, {3053353333, 1393137549, 2858266450, 954570086}},
    {{4270365148, 3284942429, 639193268, 1452194571}, {1161264254, 3727720264, 2913344122, 3397135746}},
    {{596580228, 1392092807, 843646682, 3541695628}, {2590893734, 2198545995, 237641943, 602469347}},
    {{2455879077, 2319967660, 995689268, 1471734608}, {992029320, 1234415286, 3489799059, 2101928371}}
  };
  // clang-format on
  CudaHostMem<uint4> h_output(8 * num_tasks);
  cudaMemcpy(input.Ptr(), h_input, input.SizeInBytes(), cudaMemcpyHostToDevice);
  const uint32_t BLOCK_DIM = 32;
  const uint32_t GRID_DIM = (num_tasks - BLOCK_DIM - 1) / BLOCK_DIM / 2;
  pbkdf2_128b<<<GRID_DIM, BLOCK_DIM>>>(N, starting_index, input.Ptr(),
                                       output.Ptr(), num_tasks);
  cudaMemcpy(h_output.HPtr(), output.Ptr(), h_output.SizeInBytes(),
             cudaMemcpyDeviceToHost);
  auto *p = h_output.HPtr();
  for (size_t i = 0; i < h_output_ref.size(); ++i) {
    uint4 st = p[i * 8];
    uint4 ed = p[i * 8 + 7];
    uint4 st_ref = h_output_ref[i].first;
    uint4 ed_ref = h_output_ref[i].second;
    EXPECT_TRUE(st == st_ref && ed == ed_ref);
  }
}

static __global__ void pbkdf2_32b(const ulong starting_index,
                                  const uint4 *const __restrict__ input,
                                  uint4 *const __restrict__ output,
                                  const uint num_tasks) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input[0];
    password[1] = input[1];
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_pbkdf2_32B(password, X, &output[t * 2]);
  }
}

TEST(PBKDF2_32B, CheckResult) {
  // clang-format off
  std::vector<uint4> output_ref{
    {1541574387, 488907923, 795739296, 3660924057}, {1595429963, 1442715467, 3454537610, 303168103},
    {2832999825, 1792965534, 3851604374, 1543672901}, {2252171238, 2489374518, 4196827066, 1925643828},
    {3776545153, 2493253923, 1983841114, 4189265163}, {3106590296, 3126490259, 4292680934, 3487255118},
    {1138967569, 3223815245, 3999001665, 2833458212}, {525278724, 204474235, 1590998285, 1126296421},
    {1712137169, 3649550485, 2728357207, 2572560430}, {629927608, 1494199002, 426129028, 2296958300},
    {612931869, 2928911829, 999195935, 3894654826}, {711808285, 773108236, 1884424028, 405027227},
    {3496895468, 4098261784, 1205595361, 987182193}, {1092867524, 3407967604, 1630381730, 850280901},
    {2344968929, 1230032780, 533878510, 782288479}, {3426145836, 2789001978, 2013751058, 3920530103},
    {1004256693, 2820654651, 3302594902, 1895517683}, {1715443604, 1822444432, 3642638638, 4172159742},
    {361838609, 2572724440, 3232663262, 1640532158}, {3156298490, 285734263, 2107779922, 124293624},
    {2875742700, 1740468830, 292959372, 1165028846}, {1034497710, 841845972, 3869241330, 279539866},
    {1684353161, 2791287652, 639938640, 696103378}, {1098941359, 3059816824, 80312190, 1352994082},
    {491432485, 1977606223, 568579903, 2848969311}, {856678766, 694736051, 1544114686, 3391276793},
    {528722383, 4221448681, 1427708755, 3435410113}, {108301243, 3032419668, 401350649, 1497139475},
    {1633560854, 1259258950, 2231332965, 3069488567}, {3422183957, 1262157060, 2082726213, 1043984063},
    {1659437647, 2520492023, 1526642929, 2037984020}, {3943654391, 2575523191, 3898673117, 1543536477},
  };
  // clang-format on

  uint32_t num_tasks = 8704 * 2;
  uint64_t starting_index = 0UL;
  CudaDeviceMem<uint4> input(2);
  uint4 h_input[2]{2839345266U, 42009750U,   875455879U,  2217211394U,
                   3438177526U, 2734532412U, 2819254414U, 1408356118U};
  cudaMemcpy(input.Ptr(), h_input, input.SizeInBytes(), cudaMemcpyHostToDevice);
  CudaDeviceMem<uint4> output(num_tasks * 2);
  CudaHostMem<uint4> h_out(num_tasks * 2);

  uint32_t BLOCK_DIM = 32;
  uint32_t GRID_DIM = (num_tasks - BLOCK_DIM - 1) / BLOCK_DIM / 2;
  pbkdf2_32b<<<GRID_DIM, BLOCK_DIM>>>(starting_index, input.Ptr(), output.Ptr(),
                                      num_tasks);
  cudaMemcpy(h_out.HPtr(), output.Ptr(), output.SizeInBytes(),
             cudaMemcpyDeviceToHost);
  uint4 *p = h_out.HPtr();
  for (size_t i = 0; i < output_ref.size(); ++i) {
    EXPECT_TRUE(p[i] == output_ref[i]);
  }
}

template <uint32_t LOOPUP_GAP = 2>
static __global__ void romix(const uint32_t N, const ulong starting_index,
                             const uint4 *const __restrict__ input,
                             uint4 *const __restrict__ padcache,
                             uint4 *const __restrict__ output,
                             const uint num_tasks) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t tnum = gridDim.x * blockDim.x;

  uint4 password[5];
  uint4 X[8];
  for (uint32_t t = tid; t < num_tasks; t += tnum) {
    const uint64_t index = starting_index + t;

    password[0] = input[0];
    password[1] = input[1];
    password[2].x = uint32_t(index & 0xFFFFFFFF);
    password[2].y = uint32_t((index >> 32) & 0xFFFFFFFF);
    password[2].z = 0;
    password[2].w = 0;
    password[3] = make_zero<uint4>();
    password[4] = make_zero<uint4>();

    scrypt_pbkdf2_128B(password, X);
    scrypt_ROMix_org<LOOPUP_GAP>(X, padcache, tnum, tid);
    output[t * 2] = X[0];
    output[t * 2 + 1] = X[7];
  }
}

TEST(Romix, CheckResult) {
  // clang-format off
  std::vector<std::pair<uint4, uint4>> output_ref{
    {{177095052, 2769478868, 1267860214, 3840882696}, {4209226529, 2478377306, 3668441954, 661024389}},
    {{675217897, 1210703173, 2821523351, 395671908}, {1164792708, 2377829658, 1357012668, 3086536117}},
    {{2122569980, 1521965530, 428608715, 1170231827}, {711050857, 3859067996, 201759563, 1563548156}},
    {{1157318318, 2940549691, 1127296292, 3378566105}, {4262566715, 148588802, 1200061494, 186445103}},
    {{1910618740, 3668591534, 543383426, 3111805377}, {1064989078, 4060066605, 1451448625, 391663587}},
    {{4214846473, 2831090917, 2414213101, 1567041697}, {4072116936, 1223198630, 1381754999, 3832720526}},
    {{1999710917, 3830843826, 2091719392, 3506640524}, {1612283548, 4084973170, 2005582528, 4160760954}},
    {{3701562884, 1369415191, 231608195, 3440200853}, {1936641932, 3560076262, 2893706191, 348983257}},
    {{1714814708, 806504637, 1185624757, 1309360160}, {1093527150, 3638059477, 3970998707, 2784271355}},
    {{1640097667, 3808618373, 1719302163, 3224052072}, {2464420809, 2492362386, 2238290668, 1350255901}},
    {{3593000297, 2800731264, 510712390, 3370323384}, {607950715, 1260312177, 248473348, 3713416381}},
    {{4278573785, 4219952570, 252971591, 1295396640}, {1479543003, 1443275236, 1565620974, 4172569527}},
    {{895020402, 4109313948, 184256163, 1338271264}, {2928529152, 1810204067, 3068405352, 4239813782}},
    {{4014920583, 1109162161, 4257725846, 444189643}, {1724450670, 2872480592, 4016779893, 878820321}},
    {{715651626, 3257863402, 2715797466, 2730055762}, {2101471733, 1562788948, 1551284046, 2392879513}},
    {{3992784225, 60299832, 649066318, 1383234528}, {3861971681, 1804071335, 1551305386, 1951189750}},
  };
  // clang-format on

  const uint32_t N = 8192;
  uint32_t num_tasks = 8704;
  uint64_t starting_index = 0UL;
  const uint32_t loopup_gap = 1;

  CudaDeviceMem<uint4> input(2);
  uint4 h_input[2]{2839345266U, 42009750U,   875455879U,  2217211394U,
                   3438177526U, 2734532412U, 2819254414U, 1408356118U};
  cudaMemcpy(input.Ptr(), h_input, input.SizeInBytes(), cudaMemcpyHostToDevice);
  CudaDeviceMem<uint4> output(num_tasks * 2);
  CudaHostMem<uint4> h_out(num_tasks * 2);

  uint32_t BLOCK_DIM = 32;
  uint32_t GRID_DIM = (num_tasks - BLOCK_DIM - 1) / BLOCK_DIM;
  const size_t global_size = GRID_DIM * BLOCK_DIM;
  CudaDeviceMem<uint4> loopup(N / loopup_gap * 8 * global_size);

  romix<loopup_gap><<<GRID_DIM, BLOCK_DIM>>>(
      N, starting_index, input.Ptr(), loopup.Ptr(), output.Ptr(), num_tasks);

  cudaMemcpy(h_out.HPtr(), output.Ptr(), output.SizeInBytes(),
             cudaMemcpyDeviceToHost);
  uint4 *p = h_out.HPtr();
  for (size_t i = 0; i < output_ref.size(); ++i) {
    EXPECT_TRUE(p[i * 2] == output_ref[i].first &&
                p[i * 2 + 1] == output_ref[i].second);
  }
}

TEST(SpacemeshOrg, CheckResult) {
  // clang-format off
  std::vector<std::pair<uint4, uint4>> output_ref{
    {{4048330317, 4093720124, 3952305695, 861738752}, {1207399841, 1252004780, 3769355734, 605176832}},
    {{2755521322, 1773634345, 3570559863, 3528040384}, {1426306995, 1828125550, 2362351051, 592285497}},
    {{794272509, 1869538505, 1508520130, 3189115413}, {3371395123, 664595614, 942134631, 2595808448}},
    {{2966305566, 139203859, 3236026452, 971490366}, {406197471, 2107380474, 1647245437, 1398227647}},
    {{4248325324, 2135553756, 3121138058, 2606063366}, {3712984664, 405355870, 1786205915, 2338023431}},
    {{2223254193, 320690588, 1750003793, 1980189572}, {2430931520, 2358257771, 857129483, 4174911228}},
    {{3824389675, 3900375118, 885509409, 2541713504}, {3965109822, 4266067977, 233389698, 429691579}},
    {{617654257, 287857505, 44437086, 3091373715}, {1579913930, 42500443, 741315169, 2504110009}},
    {{4185094906, 11838281, 788328873, 2513580847}, {3475313515, 1738120748, 1410721087, 3731153976}},
    {{3786242664, 575105985, 1815246426, 2602042915}, {1355784688, 2576224732, 3988570599, 858796312}},
    {{2700358536, 2813184983, 1596688001, 1714337525}, {1078584113, 3032059087, 2777935765, 1318855605}},
    {{1541547040, 1772543483, 3001615835, 2646093592}, {764031266, 303589574, 363686698, 2135502111}},
    {{1152844501, 106430079, 2715838609, 2711284980}, {3437796150, 1500472695, 2708109779, 889703778}},
    {{1770568223, 4026398494, 3075729081, 1584205826}, {2457749057, 155442015, 202804510, 2647478225}},
    {{3703726525, 394584141, 117082576, 699663196}, {1831918362, 2038557743, 2876631409, 3154481933}},
    {{2148001307, 979677799, 520145623, 64552000}, {4006002196, 3736832885, 543708505, 2041849618}},
  };
  // clang-format on

  const uint32_t N = 8192;
  uint32_t num_tasks = 8704 * 2;
  uint64_t starting_index = 0UL;
  const uint32_t loopup_gap = 2;

  uint4 h_input[2]{2839345266U, 42009750U,   875455879U,  2217211394U,
                   3438177526U, 2734532412U, 2819254414U, 1408356118U};
  CudaDeviceMem<uint4> output(num_tasks * 2);
  CudaHostMem<uint4> h_out(num_tasks * 2);

  uint32_t BLOCK_DIM = 32;
  uint32_t GRID_DIM = (num_tasks - BLOCK_DIM - 1) / BLOCK_DIM / 2;
  const size_t global_size = GRID_DIM * BLOCK_DIM;
  CudaDeviceMem<uint4> loopup(N / loopup_gap * 8 * global_size);

  scrypt_org<loopup_gap><<<GRID_DIM, BLOCK_DIM>>>(starting_index, num_tasks,
                                                  h_input[0], h_input[1],
                                                  loopup.Ptr(), output.Ptr());

  cudaMemcpy(h_out.HPtr(), output.Ptr(), output.SizeInBytes(),
             cudaMemcpyDeviceToHost);
  uint4 *p = h_out.HPtr();
  for (size_t i = 0; i < output_ref.size(); ++i) {
    EXPECT_TRUE(p[i * 2] == output_ref[i].first &&
                p[i * 2 + 1] == output_ref[i].second);
  }
}

TEST(SpacemeshAPI, CheckResult) {
  // clang-format off
  std::vector<std::pair<uint4, uint4>> output_ref{
    {{4048330317, 4093720124, 3952305695, 861738752}, {1207399841, 1252004780, 3769355734, 605176832}},
    {{2755521322, 1773634345, 3570559863, 3528040384}, {1426306995, 1828125550, 2362351051, 592285497}},
    {{794272509, 1869538505, 1508520130, 3189115413}, {3371395123, 664595614, 942134631, 2595808448}},
    {{2966305566, 139203859, 3236026452, 971490366}, {406197471, 2107380474, 1647245437, 1398227647}},
    {{4248325324, 2135553756, 3121138058, 2606063366}, {3712984664, 405355870, 1786205915, 2338023431}},
    {{2223254193, 320690588, 1750003793, 1980189572}, {2430931520, 2358257771, 857129483, 4174911228}},
    {{3824389675, 3900375118, 885509409, 2541713504}, {3965109822, 4266067977, 233389698, 429691579}},
    {{617654257, 287857505, 44437086, 3091373715}, {1579913930, 42500443, 741315169, 2504110009}},
    {{4185094906, 11838281, 788328873, 2513580847}, {3475313515, 1738120748, 1410721087, 3731153976}},
    {{3786242664, 575105985, 1815246426, 2602042915}, {1355784688, 2576224732, 3988570599, 858796312}},
    {{2700358536, 2813184983, 1596688001, 1714337525}, {1078584113, 3032059087, 2777935765, 1318855605}},
    {{1541547040, 1772543483, 3001615835, 2646093592}, {764031266, 303589574, 363686698, 2135502111}},
    {{1152844501, 106430079, 2715838609, 2711284980}, {3437796150, 1500472695, 2708109779, 889703778}},
    {{1770568223, 4026398494, 3075729081, 1584205826}, {2457749057, 155442015, 202804510, 2647478225}},
    {{3703726525, 394584141, 117082576, 699663196}, {1831918362, 2038557743, 2876631409, 3154481933}},
    {{2148001307, 979677799, 520145623, 64552000}, {4006002196, 3736832885, 543708505, 2041849618}},
  };
  // clang-format on

  uint32_t num_tasks = 8704 * 2;
  uint64_t starting_index = 0UL;

  std::vector<uint32_t> h_input{2839345266U, 42009750U,   875455879U,
                                2217211394U, 3438177526U, 2734532412U,
                                2819254414U, 1408356118U};
  std::vector<uint32_t> h_out(num_tasks * 8, 0);

  spacemesh_scrypt(0, starting_index, h_input.data(), num_tasks, h_out.data());
  uint4 *p = reinterpret_cast<uint4 *>(h_out.data());

  for (size_t i = 0; i < output_ref.size(); ++i) {
    EXPECT_TRUE(p[i * 2] == output_ref[i].first &&
                p[i * 2 + 1] == output_ref[i].second);
  }
}
TEST(SpacemeshCoalesceV1, CheckResult) {
  auto device_prop = GetDeviceProp(0);
  uint32_t block_num = device_prop.multiProcessorCount;
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t block_dim = device_prop.warpSize * smsp_num;

  const uint64_t starting_index = 0;
  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t LOOKUP_GAP = 2;
  const uint32_t thread_num = block_num * block_dim * LOOKUP_GAP;
  block_dim = 32;
  block_num = thread_num / block_dim;

  constexpr uint32_t TASK_PER_THREAD = 1;
  const uint32_t task_num = thread_num * TASK_PER_THREAD;

  CudaDeviceMem<uint4> d0_out(task_num * 2);
  CudaDeviceMem<uint4> d1_out(task_num * 2);

  const uint32_t N = 8 * 1024;
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);

  std::unique_ptr<uint4[]> h0_out(new uint4[task_num * 2]);
  std::unique_ptr<uint4[]> h1_out(new uint4[task_num * 2]);

  scrypt_org<LOOKUP_GAP><<<block_num, block_dim>>>(
      starting_index, task_num, in0, in1,
      reinterpret_cast<uint4 *>(d_lookup.Ptr()), d0_out.Ptr());

  scrypt_coalesce_access_v1<LOOKUP_GAP>
      <<<block_num, block_dim, 33 * sizeof(uint32_t) * block_dim>>>(
          starting_index, task_num, in0, in1,
          reinterpret_cast<uint32_t *>(d_lookup.Ptr()), d1_out.Ptr());

  CHECK(cudaMemcpy(h0_out.get(), d0_out.Ptr(), d0_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h1_out.get(), d1_out.Ptr(), d1_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < task_num * 2; ++i) {
    EXPECT_EQ(h0_out[i], h1_out[i]);
  }
}

TEST(SpacemeshCoalesceV2, CheckResult) {
  auto device_prop = GetDeviceProp(0);
  uint32_t block_num = device_prop.multiProcessorCount;
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t block_dim = device_prop.warpSize * smsp_num;

  const uint64_t starting_index = 0;
  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t LOOKUP_GAP = 2;
  const uint32_t thread_num = block_num * block_dim * LOOKUP_GAP;
  block_dim = 32;
  block_num = thread_num / block_dim;

  constexpr uint32_t TASK_PER_THREAD = 1;
  const uint32_t task_num = thread_num * TASK_PER_THREAD;

  CudaDeviceMem<uint4> d0_out(task_num * 2);
  CudaDeviceMem<uint4> d1_out(task_num * 2);

  const uint32_t N = 8 * 1024;
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);

  std::unique_ptr<uint4[]> h0_out(new uint4[task_num * 2]);
  std::unique_ptr<uint4[]> h1_out(new uint4[task_num * 2]);

  scrypt_org<LOOKUP_GAP><<<block_num, block_dim>>>(
      starting_index, task_num, in0, in1,
      reinterpret_cast<uint4 *>(d_lookup.Ptr()), d0_out.Ptr());

  scrypt_coalesce_access_v2<LOOKUP_GAP>
      <<<block_num, block_dim, 33 * sizeof(uint64_t) * block_dim / 2>>>(
          starting_index, task_num, in0, in1,
          reinterpret_cast<uint64_t *>(d_lookup.Ptr()), d1_out.Ptr());

  CHECK(cudaMemcpy(h0_out.get(), d0_out.Ptr(), d0_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h1_out.get(), d1_out.Ptr(), d1_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < task_num * 2; ++i) {
    EXPECT_EQ(h0_out[i], h1_out[i]);
  }
}

TEST(SpacemeshCoalesceV3, CheckResult) {
  auto device_prop = GetDeviceProp(0);
  uint32_t block_num = device_prop.multiProcessorCount;
  uint32_t smsp_num = GetSMSPNum(device_prop.major, device_prop.minor);
  uint32_t block_dim = device_prop.warpSize * smsp_num;

  const uint64_t starting_index = 0;
  uint4 in0{2839345266U, 42009750U, 875455879U, 2217211394U};
  uint4 in1{3438177526U, 2734532412U, 2819254414U, 1408356118U};

  const uint32_t LOOKUP_GAP = 2;
  const uint32_t thread_num = block_num * block_dim * LOOKUP_GAP;
  block_dim = 32;
  block_num = thread_num / block_dim;

  constexpr uint32_t TASK_PER_THREAD = 1;
  const uint32_t task_num = thread_num * TASK_PER_THREAD;

  CudaDeviceMem<uint4> d0_out(task_num * 2);
  CudaDeviceMem<uint4> d1_out(task_num * 2);

  const uint32_t N = 8 * 1024;
  CudaDeviceMem<uint32_t> d_lookup(N / LOOKUP_GAP * 32 * thread_num);

  std::unique_ptr<uint4[]> h0_out(new uint4[task_num * 2]);
  std::unique_ptr<uint4[]> h1_out(new uint4[task_num * 2]);

  scrypt_org<LOOKUP_GAP><<<block_num, block_dim>>>(
      starting_index, task_num, in0, in1,
      reinterpret_cast<uint4 *>(d_lookup.Ptr()), d0_out.Ptr());

  scrypt_coalesce_access_v3<LOOKUP_GAP>
      <<<block_num, block_dim, 33 * sizeof(uint4) * block_dim / 4>>>(
          starting_index, task_num, in0, in1,
          reinterpret_cast<uint4 *>(d_lookup.Ptr()), d1_out.Ptr());

  CHECK(cudaMemcpy(h0_out.get(), d0_out.Ptr(), d0_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h1_out.get(), d1_out.Ptr(), d1_out.SizeInBytes(),
                   cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < task_num * 2; ++i) {
    EXPECT_EQ(h0_out[i], h1_out[i]);
  }
}
