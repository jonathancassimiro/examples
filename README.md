# Examples-Codes
Replication of the results of the four mathematical examples for the RBDO process. The supporting source codes are available for download.

- The optimization used Sequential Quadratic Programming (SQP) through the Scipy library - minimize function (Scipy, 2023).
- The reliability analysis employed the First-Order Reliability Method (FORM) for RIA and PMA through the Pystra package (Hack and Caprani, 2022). SORA and SLA used a FORM routine we developed.
- The Hassofer-Lind-Rackwitz-Fiessler (HLRF) algorithm assessed the Most Probable Point (MPP) and finite differences calculated numerical derivatives.
- The jacobian of the objective function was explicitly provided to the minimize function in SciPy as an argument to improve efficiency and convergence during the minimization process.
- Convergence criteria: tolerance of objective function equal to 1E-4, max of 1,000 iterations, max of 50 cycles. 

Mathematical Examples RBDO: Nonlinear Limit State, Multiple limit states, Short Column, Gearbox, Highly Nonlinear, Hock and Schittkowski 113, Automobile with front I-beam axle, Steel T-column, Conical Structure, Tension/Compression Spring.

More informations: jonathan.cassimiro@ufpe.br
