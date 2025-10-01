from .CreateNetwork import (
    CreateGraph,
    GenerateEdgeSet,
    get_network_stats,
    save_connectivity_matrix,
)
from .ei_glm import GLMEI, projection_ei
from .io import load_model
from .Simulatenew2 import (
    Simulate,
    avgVoltage,
    step,
    trackEConductance,
    trackIConductance,
    trackVoltage,
)
from .visualizations import visualize_connectivity_matrix
from .regularizer import RidgeMultiRegularization, LassoMultiRegularization,GroupLassoMultiRegularization
from nemos._regularizer_builder import _REGULARIZER_MAP, AVAILABLE_REGULARIZERS


_REGULARIZER_MAP.update(
    {
        "infer_connectivity.regularizer.RidgeMultiRegularization": RidgeMultiRegularization,
        "infer_connectivity.regularizer.LassoMultiRegularization": LassoMultiRegularization,
        "infer_connectivity.regularizer.GroupLassoMultiRegularization": GroupLassoMultiRegularization,
        "RidgeMultiRegularization": RidgeMultiRegularization,
        "LassoMultiRegularization": LassoMultiRegularization,
        "GroupLassoMultiRegularization": GroupLassoMultiRegularization
    }
)
AVAILABLE_REGULARIZERS.extend(["RidgeMultiRegularization", "LassoMultiRegularization", "GroupLassoMultiRegularization"])
