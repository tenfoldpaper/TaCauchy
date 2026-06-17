Hi, Thank you for your contribution and hard working!

I am wondering if it is possible that I can extend the gelsight simulation to 3D deformation, like a half circle independently set in the simulation environment with the FEM for soft body and when a gripper or finger touch its surface, can I get the contact force information.

Bests.


Hi,
yeah, I think it should be possible. I didn't try it out myself, but here is some information on how to extract contact information out of the uipc simulation: spiriMirror/libuipc#163.

Just keep in mind that the gripper/finger that touch the half circle surface also have to be simulated with UIPC (there is no physical coupling between Isaac's PhysX and UIPC right now).


If you need to export Energy/Gradient/Hessian in the backend, this is what you need!

Now, regarding contact, libuipc supports classified contact energy, gradient, and Hessian.

https://github.com/spiriMirror/libuipc-samples/blob/main/python/20_contact_system_feature/main.py

Btw, in uipc, energy/gradient/hessian is scaled by $\Delta t^2$.

Cuz, the optimization problem is:
\[
E(x)
= \frac{1}{2} \lVert x - \tilde{x} \rVert_M^2
+ P(x)\,\Delta t^2
+ B(x)\,\Delta t^2
\]

where $\tilde{x}$ is the predicted position,
$P(x)$ is the elastic potential energy,
$B(x)$ is the contact potential.

So if you need the so-called \textbf{force} (from Newton's Law aspect).
\[
f = -\frac{g}{\Delta t^2}
\]

where $f$ is the force, $g$ is the gradient.
