import torch
from torch import Tensor
from typing import Optional, List, Dict, Tuple, Union
from warnings import warn
from .reference_energy import ReferenceEnergies


# for correct type conversions:
def gradient(
        y: torch.Tensor, x: List[torch.Tensor], create_graph: bool, retain_graph: bool
) -> List[torch.Tensor]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
    grads = torch.autograd.grad(
        [y],
        x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    new_grads = []
    for grad in grads:
        if grad is not None:
            new_grads.append(grad)
        else:
            raise Exception("grad was None")
    return new_grads


class BaseModel(torch.nn.Module):
    """
    Base class for implementing new models or wrapping existing models.
    """

    def __init__(
            self, hyperparameters: Dict, compute_forces: bool = False, num_datasets: int = 1
    ):
        """
        :param hyperparameters: Dictionary storing hyperparameters that are needed for saving and restoring the model
        :param compute_forces: Boolean that indicates if forces (negative gradients wrt energy) should be computed;
        default: False
        """
        super().__init__()
        self.hyperparameters = hyperparameters
        self.compute_forces = compute_forces
        self.constant_energy_model = ReferenceEnergies(
            freeze=True, max_num_datasets=num_datasets
        )

    def set_constant_energies(
            self,
            energy_dict: Dict[int, float],
            freeze: bool = True,
            dataset_index: Optional[int] = 0,
    ) -> None:
        """
        :param energy_dict: Dictionary mapping atomic numbers to atomwise energy contributions
        :param freeze: Boolean, whether atomwise energies should be frozen during training; default: True
        :return: None
        """
        self.constant_energy_model.set_constant_energies(
            energy_dict=energy_dict, freeze=freeze, dataset_index=dataset_index
        )

    def model_forward(self, data: Dict[str, Tensor]) -> Tensor:
        """
        Child classes should implement the forward pass for computing the energy here.
        :param data: Data object storing information that is need for the model
        :return: Tensor with energy prediction
        """
        raise NotImplementedError

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        torch.set_grad_enabled(True)

        forces: Optional[Tensor] = None
        if self.compute_forces:
            data["pos"].requires_grad_()

        energy = self.model_forward(data).view(-1)

        if self.compute_forces:
            grads = gradient(
                energy,
                [data["pos"]],
                create_graph=self.training,
                retain_graph=self.training,
            )

            forces = (-1.0) * grads[0]

        if "dataset_idx" in data.keys():
            dataset_idx = data["dataset_idx"]
        else:
            dataset_idx = None

        atom_energies = self.constant_energy_model(
            data["z"].long(),
            data["batch"],
            dataset_index=dataset_idx,
        )
        energy += atom_energies

        output_dict = {"energy": energy}
        if forces is not None:
            output_dict["forces"] = forces
        return output_dict
