"""GPU bridge package — split from the original monolithic ``oracle_game.gpu`` module.

Every public name is re-exported here so ``from oracle_game.gpu import X`` keeps working.
"""
from oracle_game.gpu._common import *  # noqa: F401,F403
from oracle_game.gpu.dtypes import *  # noqa: F401,F403
from oracle_game.gpu.packers import *  # noqa: F401,F403
from oracle_game.gpu.readback import *  # noqa: F401,F403
from oracle_game.gpu.bridge import *  # noqa: F401,F403

# Private helpers the original monolith exposed as module attributes.
from oracle_game.gpu._common import (  # noqa: F401
    _SHARED_STANDALONE_CONTEXT,
    _get_shared_standalone_context,
    _json_bytes,
    _json_default,
    _render_group_tile,
)
from oracle_game.gpu.packers import (  # noqa: F401
    _float_or_nan,
    _gas_ref,
    _light_ref,
    _material_ref,
    _pack_half2x16,
    _pack_pair_reaction_rules,
    _page_stripe_payload_array,
    _page_stripe_payload_key,
    _phase_mask,
    _typed_name_id,
    _unpack_half2x16,
)

