"""Official DeltaActions/AbsoluteActions around normalization, on aligned state14."""
import dataclasses
import numpy as np

from inference.real_robot.relative14 import DELTA_MASK, HORIZON, finite_width


@dataclasses.dataclass(frozen=True)
class ValidateInputs:
    def __call__(self, data):
        state = finite_width(data['state'],14,'relative14 server state')
        if state.shape != (14,):
            raise ValueError('Data transforms require one unbatched state14')
        result = {**data,'state':state.copy()}
        if 'actions' in data:
            actions = finite_width(data['actions'],14,'physical actions')
            if actions.shape != (HORIZON,14):
                raise ValueError('Expected physical H15 actions')
            # Official DeltaActions mutates actions in place; isolate dataset buffers.
            result['actions'] = actions.copy()
        return result


@dataclasses.dataclass(frozen=True)
class CropPhysical:
    def __call__(self, data):
        state, actions = np.asarray(data['state']), np.asarray(data['actions'])
        if state.shape not in ((14,),(32,)) or actions.shape not in ((15,14),(15,32)):
            raise ValueError('Unexpected model state/action shape')
        return {**data,'state':finite_width(state[:14],14,'restored state').copy(),
                'actions':finite_width(actions[:,:14],14,'relative actions').copy()}


@dataclasses.dataclass(frozen=True)
class PhysicalOutputs:
    def __call__(self, data):
        return {'actions':finite_width(data['actions'],14,'decoded physical actions').copy()}


@dataclasses.dataclass(frozen=True)
class RelativeTransforms:
    def __call__(self, model_config):
        from openpi import transforms
        return transforms.Group(
            inputs=[ValidateInputs(),transforms.DeltaActions(DELTA_MASK)],
            outputs=[CropPhysical(),transforms.AbsoluteActions(DELTA_MASK),PhysicalOutputs()])
