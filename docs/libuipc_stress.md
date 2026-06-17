Libuipc 框架下高保真力学信息提取与触觉仿真策略深度研究报告1. 绪论：物理仿真中的“黑盒”困境与力学透明化需求在现代机器人学、计算机图形学以及具身智能（Embodied AI）的研究中，基于物理的仿真（Physics-based Simulation）已经成为了不可或缺的基础设施。Libuipc（Unified Incremental Potential Contact）作为一个基于增量势能接触（IPC）理论的高性能仿真库，以其在处理刚体、软体、布料及线缆耦合时的无穿透（Penetration-free）特性和对大变形的鲁棒性而著称 。然而，对于致力于高精度触觉仿真（Tactile Simulation）或需要深入力学分析的研究者而言，现有的仿真框架往往呈现出一种“力学黑盒”的特性。具体而言，用户在使用 Python 接口进行仿真控制时，通常只能获取物体的位置（Position）、速度（Velocity）以及接触几何信息。虽然仿真器内部为了求解动力学方程，必须精确计算形变梯度（Deformation Gradient）、应变能密度（Strain Energy Density）以及应力张量（Stress Tensor），但这些核心力学量往往被视为中间变量（Intermediate Variables），在完成力的组装后即被丢弃或隐藏，未向顶层 API 暴露。本报告针对用户在 Libuipc 框架中遇到的“无法获取应力张量与内部能量梯度”这一核心问题，进行详尽的代码级架构分析、理论溯源与工程实现方案推演。分析确认，当前的 Libuipc Python 接口确实仅暴露了以几何和接触为主的特征，而未包含连续介质力学中的应力状态描述 。为了打破这一限制，支持如 GelSight、电容式压力阵列等触觉传感器的仿真，必须深入 CUDA 后端，建立从底层有限元（FEM）核函数到上层 Python 绑定的完整数据通路。本报告将分为理论基础、架构诊断、代码实现策略、触觉映射应用四个部分，系统性地阐述如何在不破坏 Libuipc 高性能架构的前提下，导出真实的柯西应力（Cauchy Stress）与内部弹性能量梯度。2. 连续介质力学与有限元理论基础在着手修改代码之前，必须明确我们需要导出的物理量的数学定义及其在有限元方法（FEM）中的计算路径。Libuipc 采用的“稳定 Neo-Hookean”（Stable Neo-Hookean）模型是一种超弹性（Hyperelastic）材料模型，理解其本构关系是正确计算应力的前提。2.1 运动学描述与形变梯度在有限元仿真中，物体的运动通过参考构型（Reference Configuration, $\Omega_0$）到当前构型（Current Configuration, $\Omega_t$）的映射 $\phi: \Omega_0 \to \Omega_t$ 来描述。对于四面体网格（Tetrahedral Mesh），最常用的单元类型是线性四面体单元（Linear Tetrahedral Element），其内部形变是均匀的。形变梯度张量（Deformation Gradient Tensor） $\boldsymbol{F}$ 是描述局部形变的核心物理量，定义为当前坐标 $\boldsymbol{x}$ 对参考坐标 $\boldsymbol{X}$ 的导数：$$\boldsymbol{F} = \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}}$$在 Libuipc 的 CUDA 实现中，$\boldsymbol{F}$ 是通过形状函数的导数计算的。对于一个四面体，$\boldsymbol{F}$ 是一个 $3 \times 3$ 的矩阵。若已知四面体四个顶点在参考构型下的坐标 $\boldsymbol{X}_0, \boldsymbol{X}_1, \boldsymbol{X}_2, \boldsymbol{X}_3$ 和当前构型下的坐标 $\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \boldsymbol{x}_3$，则 $\boldsymbol{F}$ 可通过以下矩阵运算求得：$$\boldsymbol{F} = \boldsymbol{D}_s \boldsymbol{D}_m^{-1}$$其中：$\boldsymbol{D}_s = [\boldsymbol{x}_1 - \boldsymbol{x}_0, \boldsymbol{x}_2 - \boldsymbol{x}_0, \boldsymbol{x}_3 - \boldsymbol{x}_0]$ 是变形后的形状矩阵。$\boldsymbol{D}_m = [\boldsymbol{X}_1 - \boldsymbol{X}_0, \boldsymbol{X}_2 - \boldsymbol{X}_0, \boldsymbol{X}_3 - \boldsymbol{X}_0]$ 是参考形状矩阵。由于 $\boldsymbol{D}_m^{-1}$ 仅依赖于静止状态的网格几何，通常在仿真初始化阶段预计算并存储，而 $\boldsymbol{D}_s$ 则在每一时间步实时更新 。2.2 超弹性材料与能量密度函数Libuipc 使用的 Stable Neo-Hookean 模型是为了解决传统 Neo-Hookean 模型在压缩及反转（Inversion）情况下的数值不稳定性而设计的 。其应变能密度函数 $\Psi(\boldsymbol{F})$ 通常形式如下：$$\Psi(\boldsymbol{F}) = \frac{\mu}{2} (I_C - 3) + \frac{\lambda}{2} (J - 1 - \alpha)^2$$在此公式中：$\mu$ 和 $\lambda$ 是拉梅常数（Lamé parameters），由杨氏模量 $E$ 和泊松比 $\nu$ 导出。$J = \det(\boldsymbol{F})$ 是雅可比行列式，表示体积变化比率。$I_C = \text{tr}(\boldsymbol{F}^T \boldsymbol{F})$ 是右柯西-格林张量（Right Cauchy-Green Tensor）的第一不变量。$\alpha$ 是与材料参数相关的修正项（通常用于调节静止状态下的稳定性）。该能量函数的导数决定了材料内部的恢复力。在 IPC 框架中，求解器的目标是最小化总势能，因此计算 $\Psi$ 关于位置 $\boldsymbol{x}$ 的梯度（即力）是核心步骤。2.3 应力张量的层级与物理意义在代码修改过程中，最容易混淆的是不同定义的应力张量。为了触觉仿真，我们需要明确区分以下三种张量：应力类型符号定义域物理意义代码对应变量第一类 Piola-Kirchhoff 应力$\boldsymbol{P}$参考构型 (Material)单位参考面积上的当前力dEdF第二类 Piola-Kirchhoff 应力$\boldsymbol{S}$参考构型 (Material)单位参考面积上的参考力dEdE 相关柯西应力 (Cauchy Stress)$\boldsymbol{\sigma}$当前构型 (Spatial)单位当前面积上的当前力需计算关键洞察：仿真器内部使用 $\boldsymbol{P}$：因为 FEM 积分通常在参考体积上进行，计算节点力 $\boldsymbol{f}$ 的公式为 $\boldsymbol{f} = -V_{ref} \boldsymbol{P} (\nabla N)^T$。因此，Libuipc 的底层代码必然显式计算了 $\boldsymbol{P}$ 。传感器感知使用的是 $\boldsymbol{\sigma}$：触觉传感器（如皮肤、压力垫）位于物理世界中，感知的是当前构型下的压强分布。因此，我们不能直接导出 dEdF ($\boldsymbol{P}$)，而必须将其转换为 $\boldsymbol{\sigma}$。转换公式为：$$\boldsymbol{\sigma} = \frac{1}{J} \boldsymbol{P} \boldsymbol{F}^T$$这意味着，在获得 dEdF 和 F 之后，必须进行一次矩阵乘法和标量除法，才能得到对用户有意义的物理量 。2.4 关于 betterforce 的“normal_pressure / tangential_pressure”需澄清：这两者不是材料内部应力，而是 IPC 接触势垒梯度 contact_gradient 经 dt^2 还原成接触力后，再除以受支配面积得到的表面牵引（traction）。其核心公式为 p = f / A_tributary，随后再分解为法向与切向分量。由于 IPC 势垒梯度在接触边缘更先激活并呈尖峰，即使做了面积归一化与法线平滑，仍可能出现“边缘高、中心低”的伪分布。因此 betterforce 只能视为数值接触牵引的近似，而非真实压力/应力。3. Libuipc 架构深度剖析与断点定位基于用户提供的代码审查结论及对 Libuipc 代码结构的推断，我们对系统后端进行“外科手术式”的定位。3.1 现有数据流分析Libuipc 的执行流程可以概括为：Python 层：设置场景，调用 world.step()。C++ Host 层：调度器调用 FiniteElementMethod::compute_gradient_hessian()。CUDA Device 层：启动内核（Kernel），并行计算每个四面体的能量和梯度。在文件 libuipc/src/backends/cuda/finite_element/constitutions/stable_neo_hookean_3d.cu 中，计算逻辑如下：C++// 伪代码流程重建
__device__ void compute_element_gradient(...) {
    // 1. 读取节点位置 x
    // 2. 加载预计算的 Dm_inv
    auto F = fem::F(x0, x1, x2, x3, Dm_inv); 
    
    // 3. 计算能量导数 (即 PK1 应力 P)
    // SNH::dEdVecF 计算的是 dPsi/dF，直接对应 P
    SNH::dEdVecF(dEdF, mu, lambda, VecF); 
    
    // 4. 将 P 转换为节点力 G (Internal Energy Gradient)
    Matrix9x12 dFdx = fem::dFdx(Dm_inv);
    Vector12 G = dFdx.transpose() * dEdF; 
    
    // 5. 原子加法汇编到全局力向量
    atomicAdd(global_force, G);
    
    // [断点]：在此处，F 和 dEdF (P) 均作为寄存器变量存在。
    // 计算完成后，函数返回，这些值被丢弃。
}
3.2 缺失的环节用户指出的“没有 pybind 绑定”是表象，根本原因在于数据生命周期。stress (应力) 和 strain (应变) 不是系统的状态量（State Variables）。系统的状态仅由位置 $x$ 和速度 $v$ 决定。应力是派生量（Derived Quantities），仅在力学求解的瞬间存在于 GPU 寄存器中。现有的 ContactSystemFeature.contact_gradient 之所以能获取，是因为接触力往往需要用于后续的摩擦计算或作为单独的 Debug 信息被显式存储了。而弹性内力通常直接累加到总合力中，没有单独的缓冲区。3.3 解决方案的可行性评估用户提出的两种方案：C++/CUDA 层导出接口：这是最根本的解决之道。必须在内核中开辟一段显存（Global Memory），将寄存器中的 F 和 dEdF 转化为 $\boldsymbol{\sigma}$ 并写入。现有 FEM 组件中找 hook：由于 Libuipc 的高性能设计倾向于 kernel fusion（核融合）以减少显存带宽，不太可能预留了未使用的“观察者模式”接口。因此，必须修改源码。推荐策略：基于属性（Attribute-based）的导出方案（方案 B）。
Libuipc 的几何表达（SimplicialComplex）支持动态属性。利用这一机制，我们可以在四面体（Tetrahedra）上挂载一个名为 cauchy_stress 的 $3 \times 3$ 矩阵属性。这不仅符合 Libuipc 的设计哲学，也方便通过现有的几何接口在 Python 端读取，无需重写大量底层传输代码 。4. 核心实现策略：从 CUDA 内核到 Python 接口本章详述实现步骤。这将涉及 CUDA 内核修改、C++ 宿主代码更新以及 Pybind11 接口暴露。4.1 第一步：在 C++ 层注册应力属性在 FiniteElementMethod 或具体的材料模型类（如 FEM3DConstitution）初始化阶段，需要请求几何模块分配额外的显存空间。代码逻辑修改位置： libuipc/src/backends/cuda/finite_element/finite_element_method.cu (或类似初始化文件)C++// 在初始化 FEM 系统时
void FiniteElementMethod::build(...) {
    //... 现有初始化代码...
    
    // 新增：检查是否开启应力导出配置
    if (config.export_stress) {
        // 在四面体集合上创建名为 "stress" 的属性
        // 假设 geometry().tetrahedra() 返回的是几何容器
        // create_attribute<Matrix3x3> 会在 GPU 上分配连续内存
        m_stress_attr = geometry().tetrahedra().create_attribute<Matrix3x3>("stress");
    }
}
4.2 第二步：修改 CUDA 内核计算并存储柯西应力这是最关键的一步。我们需要进入 stable_neo_hookean_3d.cu 或调用它的上层 Kernel，将应力计算逻辑嵌入。代码逻辑修改位置： libuipc/src/backends/cuda/finite_element/finite_element_energy_producer.cu算法实现细节：C++// 假设这是运行在 GPU 上的 Kernel 函数
__global__ void compute_gradient_kernel(
    //... 原有参数...
    Matrix3x3* stress_buffer // 新增参数：指向应力属性的指针
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    //... 原有计算 F 和 dEdF 的代码...
    // auto F =...;
    // auto dEdF =...; // 这是 PK1 应力 P

    // === 新增：应力导出逻辑 ===
    if (stress_buffer!= nullptr) {
        // 1. 计算雅可比行列式 J
        Scalar J = F.determinant();

        // 2. 防止数值爆炸：如果 J 极小，应力可能会变得非物理的巨大
        // 在 Stable Neo-Hookean 中，通常 J > 0 是受约束保证的，但为了安全可加 epsilon
        if (J < 1e-6) J = 1e-6; 

        // 3. 计算柯西应力: sigma = (1/J) * P * F^T
        // dEdF 是 3x3 矩阵 (reshaped from vector if needed)
        // 注意：Eigen 或 Libuipc 的矩阵乘法语法
        Matrix3x3 P = dEdF; 
        Matrix3x3 sigma = (1.0 / J) * P * F.transpose();

        // 4. 写入全局显存
        stress_buffer[i] = sigma;
    }

    //... 原有力的组装代码...
}
深入洞察：这里存在一个性能权衡（Trade-off）。写入全局显存（Global Memory Store）会消耗显存带宽。对于数百万网格的仿真，这可能会导致求解器性能下降 5%-10%。因此，建议将 stress_buffer 指针设为可选（Optional），仅在用户显式开启 export_stress=True 时传入有效指针，否则传入 nullptr 以跳过写入分支。4.3 第三步：Python 侧的 Pybind11 暴露一旦数据位于 GPU 的 Attribute 中，我们需要确保 Python 可以访问它。Libuipc 的 Python 绑定通常位于 src/pybind。代码逻辑修改位置： libuipc/src/pybind/pyuipc/geometry/simplicial_complex.cpp用户指出没有找到 stress 相关的绑定，这是因为属性是动态的。我们需要确保 SimplicialComplex 类暴露了一个通用的 find_attribute 接口，或者专门为四面体属性添加接口。Pybind11 实现建议 ：C++// 在 SimplicialComplex 的 pybind 定义中
py::class_<SimplicialComplex>(m, "SimplicialComplex")
    //... 现有定义...
   .def("get_tet_stress",(SimplicialComplex& self) {
        // 1. 查找属性
        auto attr = self.tetrahedra().find_attribute<Matrix3x3>("stress");
        if (!attr) throw std::runtime_error("Stress attribute not found. Did you enable export_stress?");
        
        // 2. 将数据从 GPU 同步到 CPU (或者返回 GPU tensor view 如果支持)
        // 假设 attr->view() 返回 host 端的 std::vector 或类似结构
        std::vector<Matrix3x3> data = attr->download(); 
        
        // 3. 转换为 NumPy 数组返回
        // 形状: (N_tets, 3, 3)
        return py::array_t<Scalar>(
            {static_cast<ssize_t>(data.size()), 3, 3}, // Shape
            {sizeof(Matrix3x3), sizeof(Scalar)*3, sizeof(Scalar)}, // Strides
            reinterpret_cast<Scalar*>(data.data()) // Pointer
        );
    });
关键技术点：Zero-Copy vs. Safety如果直接返回 GPU 显存指针给 Python（通过 DLPack 或 torch.as_tensor），可以实现零拷贝，极大地提高后续深度学习处理的速度。如果 Libuipc 的 Python 接口主要基于 NumPy，则必须进行 Device -> Host 的拷贝。对于 10万个四面体，数据量约为 $10^5 \times 9 \times 4$ bytes $\approx 3.6$ MB，拷贝延迟在毫秒级，通常可以接受。5. 求解可靠力学信息的策略：从四面体应力到表面触觉成功导出四面体应力张量（Tetrahedral Stress Tensor）只是第一步。在触觉仿真（Tactile Simulation）中，传感器（如 GelSight 或电子皮肤）通常位于物体表面，且感知的是接触面压强（Contact Pressure）或表面牵引力（Surface Traction）。如何将体素化（Volumetric）的 $\boldsymbol{\sigma}$ 映射到表面，是连接 FEM 与触觉感知的核心算法。5.1 维度降维难题数据源：四面体内部的柯西应力 $\boldsymbol{\sigma}_{tet}$（常数张量，Constant per element）。目标：表面三角形上的牵引力矢量 $\boldsymbol{t}$ 或标量压强 $p$。5.2 投影算法（Projection Algorithm）根据柯西应力原理（Cauchy's Stress Principle），经过表面法向量 $\boldsymbol{n}$ 的平面上的牵引力向量 $\boldsymbol{t}$ 定义为：$$\boldsymbol{t} = \boldsymbol{\sigma} \cdot \boldsymbol{n}$$具体实施步骤：表面提取：在 Python 端，利用 SimplicialComplex 的拓扑信息，识别出属于边界（Boundary）的三角形面片。法向量计算：计算每个边界三角形在当前构型下的单位法向量 $\boldsymbol{n}$。注意，必须使用变形后（Deformed）的顶点坐标 $\boldsymbol{x}$ 来计算，而非参考坐标 $\boldsymbol{X}$。$$\boldsymbol{n} = \frac{(\boldsymbol{x}_1 - \boldsymbol{x}_0) \times (\boldsymbol{x}_2 - \boldsymbol{x}_0)}{\| (\boldsymbol{x}_1 - \boldsymbol{x}_0) \times (\boldsymbol{x}_2 - \boldsymbol{x}_0) \|}$$单元关联（Element Retrieval）：对于每一个边界三角形，找到其归属的四面体 ID。由于四面体是实体的，边界三角形必然是某个四面体的一个面。张量投影与分解：读取该四面体的柯西应力 $\boldsymbol{\sigma}_{tet}$，计算牵引力 $\boldsymbol{t} = \boldsymbol{\sigma}_{tet} \boldsymbol{n}$。进一步分解为：法向压强（Normal Pressure）：$p = \boldsymbol{t} \cdot \boldsymbol{n} = \boldsymbol{n}^T \boldsymbol{\sigma}_{tet} \boldsymbol{n}$。通常 $p < 0$ 表示压力（Pressure），$p > 0$ 表示张力（Tension）。触觉传感器主要关注 $p < 0$ 的部分。切向摩擦力（Shear Stress）：$\boldsymbol{\tau} = \boldsymbol{t} - (\boldsymbol{t} \cdot \boldsymbol{n})\boldsymbol{n}$。这对检测滑动（Slip detection）至关重要 。5.3 改进：节点平均法（Nodal Averaging）直接使用四面体常数应力会导致在网格边界处出现应力不连续（Stress Discontinuity），生成的触觉图像会有明显的“马赛克”效应。优化方案：外推至节点：利用体积加权平均（Volume-weighted Average）或其他平滑技术（如 Superconvergent Patch Recovery, SPR），将周围四面体的应力平均到共享顶点上，得到节点应力张量 $\boldsymbol{\sigma}_{node}$。插值：在三角形面上，利用重心坐标（Barycentric Coordinates）对节点应力进行插值，获得连续的应力场分布。高分辨率渲染：对于光学触觉传感器（如 GelSight），可以在插值后的连续场上进行高分辨率采样，生成逼真的高度图或力场图 。6. 下游应用：基于应力张量的触觉仿真与学习一旦打通了上述数据链路，Libuipc 将不仅是一个动力学求解器，更是一个高保真的多模态数据生成器。6.1 光学触觉传感器（GelSight/Digit）仿真现有的 TacEx 或 DiffTactile 等仿真器通常依赖于接触位置的几何穿透深度来估算力 。这种方法在模拟软-软接触（Soft-body to Soft-body contact）时误差极大。
通过导出 $\boldsymbol{\sigma}$，我们可以驱动更精确的标记点位移场（Marker Motion Field）模型：模型：利用弹性理论，标记点的位移 $\boldsymbol{u}_{marker}$ 与局部剪切应力 $\boldsymbol{\tau}$ 和法向压强 $p$ 直接相关。优势：能够捕捉到几何穿透无法描述的切向扭曲（Twisting）和摩擦粘滞-滑动（Stick-Slip）现象。6.2 强化学习（RL）的状态空间增强在机械手抓取任务的 RL 训练中，直接将 $3 \times 3$ 的平均抓取应力张量作为 Observation 输入，比单纯的接触布尔值包含更多信息：抓取稳定性：通过监测剪切分量与法向分量的比率（摩擦圆锥约束），Agent 可以预判物体是否即将滑落。材质识别：应力与形变的比例关系直接反映了物体的刚度，有助于 Agent 在交互中识别物体材质 。7. 结论与实施路线图通过对 Libuipc 代码库和有限元理论的深入分析，我们确认了“应力张量”与“内部能量梯度”在当前 Python 接口中的缺失并非物理计算的缺失，而是数据暴露层级的阻断。解决这一问题的关键在于利用 Libuipc 灵活的几何属性系统，在 CUDA 内核中拦截瞬时计算的应力变量。建议的实施路线图（Action Plan）：Day 1: C++ 属性注册在 FiniteElementMethod 初始化逻辑中，添加 stress 属性的创建代码。确保该属性分配在 GPU 设备内存中。Day 2: CUDA 内核注入修改 stable_neo_hookean_3d.cu 中的梯度计算函数。实现 $\boldsymbol{\sigma} = J^{-1} \boldsymbol{P} \boldsymbol{F}^T$ 的计算逻辑。添加条件判断，仅在 stress_ptr!= nullptr 时执行写入，避免无谓的性能损耗。Day 3: Pybind11 接口打通在 simplicial_complex.cpp 中添加 get_attribute 辅助函数。实现 GPU std::vector 到 NumPy ndarray 的高效转换。Day 4: Python 端验证与应用编写单元测试：模拟一个被压缩的立方体，验证导出的 $\sigma_{zz}$ 是否与理论值 $E \cdot \epsilon$ 吻合。编写工具函数 compute_surface_traction(mesh, stress)，为触觉仿真提供直接支持。通过上述步骤，可以将 Libuipc 从一个单纯的“位置求解器”升级为具备“力学感知能力”的仿真平台，为精细操作（Dexterous Manipulation）和 Sim-to-Real 研究提供坚实的数据支撑。附录：对比数据表特征现有 Python 接口 (Current)改造后目标 (Target)主要用途几何信息顶点位置 x (N,3)顶点位置 x (N,3)视觉渲染力学信息接触势能梯度 contact_gradient柯西应力张量 stress (M,3,3)触觉仿真、强度分析能量信息总系统能量 E单元内部能量梯度 internal_force稳定性分析数据流向GPU -> CPU (仅位置)GPU -> CPU (位置 + 应力)多模态学习本报告确认了用户关于 Libuipc 接口限制的结论，并提供了基于底层代码修改的完整解决方案，涵盖了从理论推导到工程落地的全链路分析。

8. 面向真实接触表面压力的工程实施方案（结合现有工程结构）

目标：输出接触表面真实压力（Cauchy stress 投影），避免 IPC 牵引的边缘伪影。

8.1 FEM 后端导出 Cauchy 应力张量（tet 级）
1) 在材料内核（如 stable_neo_hookean_3d.cu）中保留 F 与 dEdF，并计算：
   σ = (1/J) * P * F^T，其中 P = dEdF, J = det(F)。
2) 将 σ 写入 tet 级 buffer 或 tetrahedra attribute（stress）。
3) 通过 pybind 暴露 stress 读取接口（CPU copy 或 DLPack）。

8.2 表面牵引与压力投影（Python 层）
1) 取当前 deform 后的表面三角形（boundary triangles），并计算法向 n。
2) 对每个表面三角形，定位其所属 tet，读取 σ_tet。
3) 计算表面牵引 t = σ_tet · n。
4) 计算真实法向压力 p = n^T σ_tet n，切向剪应力 τ = t - p n。

8.3 节点/像素层的平滑与可视化
1) 对表面应力做体积加权的节点平均（Volume-weighted average）。
2) 在表面三角上做重心插值，得到连续压力场。
3) 投影到传感器网格（20×20 或更高）并高斯平滑。

8.4 与当前工程对接点（建议落地路径）
- 后端：stable_neo_hookean_3d.cu 中计算 σ 并写入 tet attribute。
- Python：在 tacex_uipc 层增加 StressExtractor（读取 tet stress → surface traction）。
- 可视化：复用 ContactForceGridVisualizer，用真实 p 替代 betterforce p。

8.5 为什么该方案能消除边缘伪影
- σ 来自材料内部应变能梯度，不依赖 IPC barrier 约束集。
- 牵引 t = σ·n 为连续物理量，接触中心应呈最大压力。
- 通过节点平均/插值可避免网格边界锯齿。