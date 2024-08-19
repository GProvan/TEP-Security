import torch
import pandas as pd
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
class PrincipalAngleLoss(nn.Module):
	"""
	A custom PyTorch loss function that calculates the principal angle between
	the reconstructed data and the original data in an autoencoder. The loss
	function encourages the autoencoder to learn a representation that is
	similar to the original data in terms of the principal angles between the
	data points.

	The computation of the principal angle is more computationally efficient than using a Hankel matrix,
	singular value decomposition (SVD), and then the principal angle for several reasons:
	1. Avoiding the Hankel Matrix Construction: The Hankel matrix construction involves creating a
	matrix from the data by concatenating its shifted versions. This can be computationally expensive
	for large datasets.

	2. Avoiding the SVD: The singular value decomposition (SVD) of a matrix involves finding its
	eigenvalues and eigenvectors. This can be computationally expensive, especially for large matrices.

	3. Using dot products instead of eigenvalues and eigenvectors: In this class, the cosine of the
	principal angle is computed using the dot product of the normalized vectors, which is more
	computationally efficient than computing the angle using eigenvalues and eigenvectors.

	4. Using vector operations instead of matrix operations: This class performs operations on
	individual vectors instead of matrices, which is more computationally efficient.


	Parameters:
	None

	Attributes:
	None
	"""
	
	def __init__(self):
		"""
		Initializes the PrincipalAngleLoss class by calling the super class
		constructor.
		"""
		super(PrincipalAngleLoss, self).__init__()
	
	def forward(self, recon_x, x):
		"""
		Computes the principal angle loss between the reconstructed data and the
		original data. The input data (recon_x and x) are first normalized by
		dividing by their norms. The dot product of the two normalized vectors
		is calculated and then used to compute the cosine of the principal angle
		between the two vectors. The principal angle is then calculated using
		the acos function, and the mean of the angles is used as the loss value.

		Parameters:
		recon_x (Tensor): The reconstructed data.
		x (Tensor): The original data.

		Returns:
		loss (Tensor): The principal angle loss between the reconstructed data
			and the original data.
		"""
		recon_x = recon_x / recon_x.norm(dim=1, keepdim=True)
		x = x / x.norm(dim=1, keepdim=True)
		recon_x_t = recon_x.t()
		dot_prod = torch.mm(recon_x, x.t())
		cos_theta = dot_prod.diagonal(dim1=-2, dim2=-1)
		theta = torch.acos(cos_theta)
		loss = theta.mean()
		return loss


class OneClassSVMLoss(nn.Module):
	"""
	Implementation of the One-Class SVM Loss for unsupervised anomaly detection.
	The loss is calculated based on the reconstruction error between the input and its reconstructed output.

	This implementation calculates the One-Class SVM Loss based on the mean reconstruction error between
	the input and its reconstructed output. The loss is then regularized by a parameter nu that determines
	the trade-off between the maximum mean reconstruction error and the number of violations.
	The mean reconstruction error is calculated by taking the mean of the squared differences between
	recon_x and x along each sample. The negative samples are then clamped to 0 if they are less than 1,
	and the final loss is calculated by taking the sum of the negative samples and multiplying it by nu
	divided by the number of samples.
	"""
	
	def __init__(self, nu=0.1):
		"""
		Initialization of the One-Class SVM Loss class.
		:param nu: The regularization parameter that determines the trade-off between the maximum mean reconstruction error and the number of violations.
		"""
		super(OneClassSVMLoss, self).__init__()
		self.nu = nu
	
	def forward(self, recon_x, x):
		"""
		Calculates the One-Class SVM Loss.
		:param recon_x: The reconstructed output of the autoencoder.
		:param x: The original input to the autoencoder.
		:return: The calculated loss.
		"""
		scores = torch.mean((recon_x - x) ** 2, dim=1)  # Calculate the mean reconstruction error along each sample
		negative_samples = scores
		negative_loss = torch.clamp(negative_samples + 1, min=0).sum()  # Clamp all scores that are less than 1 to 0
		loss = self.nu * negative_loss / scores.size(0)  # Calculate the final loss
		return loss


class CosineLoss(nn.Module):
	"""Cosine Loss module to compute the cosine similarity between two input tensors.
 
	Args:
		dim (int, optional): The dimension along which the cosine similarity will be computed. Default is 1.
		eps (float, optional): Small value added to the denominator to prevent division by zero. Default is 1e-8.
 
	Returns:
		Tensor: The computed cosine loss.
	"""
	
	def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
		super(CosineLoss, self).__init__()
		self.dim = dim
		self.eps = eps
	
	def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
		"""Computes the cosine similarity between two input tensors.
 
		Args:
			x1 (torch.Tensor): First input tensor.
			x2 (torch.Tensor): Second input tensor.
 
		Returns:
			Tensor: The cosine similarity loss between the two input tensors.
		"""
		similarity = F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)
		loss = 1 - torch.mean(similarity)
		return loss
