cloudViewer.pipelines.registration
----------------------------------

.. currentmodule:: cloudViewer.pipelines.registration

.. automodule:: cloudViewer.pipelines.registration

**Classes**

.. autosummary::

    CauchyLoss
    CorrespondenceChecker
    CorrespondenceCheckerBasedOnDistance
    CorrespondenceCheckerBasedOnEdgeLength
    CorrespondenceCheckerBasedOnNormal
    FastGlobalRegistrationOption
    Feature
    GMLoss
    GlobalOptimizationConvergenceCriteria
    GlobalOptimizationGaussNewton
    GlobalOptimizationLevenbergMarquardt
    GlobalOptimizationMethod
    GlobalOptimizationOption
    HuberLoss
    ICPConvergenceCriteria
    L1Loss
    L2Loss
    PoseGraph
    PoseGraphEdge
    PoseGraphNode
    RANSACConvergenceCriteria
    RegistrationResult
    RobustKernel
    TransformationEstimation
    TransformationEstimationForColoredICP
    TransformationEstimationForGeneralizedICP
    TransformationEstimationPointToPlane
    TransformationEstimationPointToPoint

**Functions**

.. autosummary::

    compute_fpfh_feature
    correspondences_from_features
    evaluate_registration
    get_information_matrix_from_point_clouds
    global_optimization
    registration_colored_icp
    registration_fgr_based_on_correspondence
    registration_fgr_based_on_feature_matching
    registration_generalized_icp
    registration_icp
    registration_ransac_based_on_correspondence
    registration_ransac_based_on_feature_matching

.. toctree::
    :hidden:

    Feature <cloudViewer.pipelines.registration.Feature>
    ICPConvergenceCriteria <cloudViewer.pipelines.registration.ICPConvergenceCriteria>
    RANSACConvergenceCriteria <cloudViewer.pipelines.registration.RANSACConvergenceCriteria>
    RegistrationResult <cloudViewer.pipelines.registration.RegistrationResult>
    registration_icp <cloudViewer.pipelines.registration.registration_icp>
    registration_colored_icp <cloudViewer.pipelines.registration.registration_colored_icp>
    evaluate_registration <cloudViewer.pipelines.registration.evaluate_registration>
