from uqgrid.models.genrou_imp import GenGENROU


class GenGENSAL(GenGENROU):
    """
    Salient-pole synchronous generator model.

    Parameters follow PSS/E ordering:
    T_d0p, T_d0dp, T_q0dp, H, D, x_d, x_q, x_dp, x_ddp, xl, S1, S2.
    """

    def __init__(
        self,
        id_tag,
        x_d,
        x_q,
        x_dp,
        x_ddp,
        xl,
        H,
        D,
        T_d0p,
        T_d0dp,
        T_q0dp,
        S1=0.0,
        S2=0.0,
    ):
        super().__init__(
            id_tag,
            x_d,
            x_q,
            x_dp,
            x_dp,
            x_ddp,
            xl,
            H,
            D,
            T_d0p,
            T_d0p,
            T_d0dp,
            T_q0dp,
            S1,
            S2,
        )
