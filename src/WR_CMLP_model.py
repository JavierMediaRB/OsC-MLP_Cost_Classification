# -*- coding: utf-8 -*-
"""
@author: Javier Mediavilla Relaño
"""
import matplotlib.pyplot
import numpy
import sklearn.metrics
import torch
import tqdm
import typing


class WR_CMLP_model(torch.nn.Module):
    """
    This class implements the tools to train and make predictions with the WR-CMLP method implemented in pytorch.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initialization of the Neural Network for the specified size.

        Args:
            input_size (int): The number of dimensions of the input data.
            hidden_size (int): The number of hidden neurons on the NN model.
        """

        super(WR_CMLP_model, self).__init__()

        self.n_out = 1
        self.a0 = torch.from_numpy(numpy.array(1., dtype=numpy.float)).double()
        self.a1 = torch.from_numpy(numpy.array(1., dtype=numpy.float)).double()

        def init_weights(m):
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)

        self.hidden0 = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size))  # First layer of NN
        self.hidden0.apply(init_weights)  # weights initialization

        self.out = torch.nn.Sequential(torch.nn.Linear(hidden_size, self.n_out))  # Second layer of NN
        self.out.apply(init_weights)  # weights initialization

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        This function evaluates the Neural Network model

        Args:
            x_input (torch.Tensor): The input data to feed the model

        Returns:
            o (torch.Tensor): The output of the model

        """

        o = torch.relu(self.hidden0(x_input))  # Activation for hidden layers
        o = torch.tanh(self.out(o))  # Activation for output layer

        return o

    def predict_class(self, model_output: torch.Tensor, q_neutral: float) -> torch.Tensor:
        """
        This method applies a decision threshold on a vector of model outputs.

        The key aspect of this threshold is that the costs are not necesary, so can make decisions
        on production data with unseen costs.

        Args:
            model_output (torch.Tensor): A torch tensor with the pre-evaluated model outputs for the data.
            q_neutral (float): The amount of neutral weighted rebalancing during training on the loss function.

        Returns:
            decisions (torch.Tensor): The decided class for each sample
        """

        model_output_aux = model_output.view(len(model_output))
        nu = q_neutral * self.a0 / self.a1
        frontier = torch.div((nu.double() - 1), (nu.double() + 1))
        y_pred = model_output_aux.double() > frontier.double()
        decisions = (2 * y_pred.double()) - 1

        return decisions

    def estimate_a(self,
                   target: torch.Tensor,
                   cr01: torch.Tensor,
                   cr10: torch.Tensor,
                   cr00: torch.Tensor,
                   cr11: torch.Tensor,
                   verbose=False
                   ) -> None:
        """
        This function estimates the value of the normalization terms a0 and a1.

        Args:
            target (torch.Tensor): The target of the samples used to estimate a0 and a1
            cr10 (torch.Tensor): The cost C10 for the train samples
            cr01 (torch.Tensor): The cost C01 for the train samples
            cr00 (torch.Tensor): The cost C00 for the train samples
            cr11 (torch.Tensor): The cost C11 for the train samples
            verbose (bool): If True, the values of the computed a0 and a1 are displayed.
        """

        pos0 = target == -1
        d0 = cr10 - cr00
        d1 = cr01 - cr11
        self.a0 = torch.mean(d0[pos0])  # Average sample estimator
        self.a1 = torch.mean(d1[~pos0])  # Average sample estimator

        if verbose is True:
            print('self.a0 ->', self.a0)
            print('self.a1 ->', self.a1)
            print('Division ->', self.a1/self.a0)

    def neutral_cost_weighted_alphas(self,
                                     target: torch.Tensor,
                                     c01: torch.Tensor,
                                     c10: torch.Tensor,
                                     c00: torch.Tensor,
                                     c11: torch.Tensor,
                                     rebalance: float
                                     ) -> typing.List[torch.Tensor]:
        """
        This function computes the value of the cost weighting applied to the training loss function.

        Args:
            target (torch.Tensor): The target of the samples
            c01 (torch.Tensor): The cost C01 for the samples
            c10 (torch.Tensor): The cost C10 for the samples
            c00 (torch.Tensor): The cost C00 for the samples
            c11 (torch.Tensor): The cost C11 for the samples
            rebalance (float): The rebalance applied to the training.

        Returns:
            alphas_cost (torch.tensor): The alpha for the costs
            alphas_RB (torch.tensor): The alpha for the rebalance
        """

        pos0 = target == -1

        # ##### alpha with Costs ##### #
        d0 = c10 - c00
        d1 = c01 - c11

        alpha0_cost = torch.div(d0, self.a0)
        alpha1_cost = torch.div(d1, self.a1)

        alphas_cost = alpha1_cost
        alphas_cost[pos0] = alpha0_cost[pos0]
        # ############################ #

        # ##### alpha with Rebalance ##### #
        alpha0_RB = torch.ones(len(c10)) * (1 / rebalance)
        alpha1_RB = torch.ones(len(c10))

        alphas_RB = alpha1_RB
        alphas_RB[pos0] = alpha0_RB[pos0]
        # ################################ #

        # ##### Resize ##### #
        alphas_cost = alphas_cost.view(len(target), 1)
        alphas_RB = alphas_RB.view(len(target), 1)
        # ################## #

        return alphas_cost, alphas_RB

    def weighted_cost_norm_mse_loss(self,
                                    model_output: torch.Tensor,
                                    target: torch.Tensor,
                                    alpha_cost: torch.Tensor,
                                    alpha_RB: torch.Tensor
                                    ) -> torch.Tensor:
        """
        This function computes the proposed cost and weighted loss to train the NN model

        Args:
            output (torch.Tensor): The output of the evaluated model
            target (torch.Tensor): The target of the samples
            alpha_cost (torch.Tensor): The alpha for the costs
            alpha_RB (torch.Tensor): The alpha for the rebalance

        Returns:
            cost_loss (torch.Tensor): The cost loss to train the NN model.
        """

        cost_loss = torch.div(torch.sum(alpha_RB * alpha_cost * ((model_output - target) ** 2)), torch.sum(alpha_RB))

        return cost_loss

    def train_discriminator(self,
                            optimizer: typing.Callable,
                            criterion: typing.Callable,
                            num_epochs: int,
                            batch_size: int,
                            inputs_train: torch.Tensor,
                            labels_train: torch.Tensor,
                            inputs_test: torch.Tensor,
                            labels_test: torch.Tensor,
                            cr01: torch.Tensor,
                            cr10: torch.Tensor,
                            cr00: torch.Tensor,
                            cr11: torch.Tensor,
                            cs01: torch.Tensor,
                            cs10: torch.Tensor,
                            cs00: torch.Tensor,
                            cs11: torch.Tensor,
                            ir_objetive: float = 1.0,
                            ir_original: float = 1.0,
                            figure: bool = False,
                            metrics: bool = True,
                            ) -> typing.List[numpy.ndarray]:
        """
        This function trains the WR-CMLP method and plots some optional graphics and metrics to measure the performance
        and the behaviour of the training process.

        Args:
            optimizer (typing.Callable): The optimizer to train the weights of the NN model.
            criterion (typing.Callable): The training loss function
            num_epochs (int): The number of epochs of the training process. Also to perform manual early stopping.
            batch_size (int): The number of samples to include in each batch
            inputs_train (torch.Tensor): The input train samples
            labels_train (torch.Tensor): The target labels for the train samples
            inputs_test (torch.Tensor): The input train samples
            labels_test (torch.Tensor): The target labels for the test samples
            cr01 (torch.Tensor): The cost C01 for the train samples
            cr10 (torch.Tensor): The cost C10 for the train samples
            cr00 (torch.Tensor): The cost C00 for the train samples
            cr11 (torch.Tensor): The cost C11 for the train samples
            cs01 (torch.Tensor): The cost C01 for the test samples
            cs10 (torch.Tensor): The cost C10 for the test samples
            cs00 (torch.Tensor): The cost C00 for the test samples
            cs11 (torch.Tensor): The cost C11 for the test samples
            ir_objetive (float): Default to 1.0. The objective imbalance after applying the rebalance.
            ir_original (float): Default to 1.0: The original imbalance of the dataset.
            figure (bool): Default to False. If True, display a figure with the behaviour of the training process.
            metrics (bool): Default to True. If True, compute the metrics of the model for each epoch of the training.

        Returns:
            loss_train (numpy.ndarray): A vector with the train loss value along the epoch of the training
            loss_test (numpy.ndarray): A vector with the test loss value along the epoch of the training
            saving_train (numpy.ndarray): A vector with the train savings value along the epoch of the training
            saving_test (numpy.ndarray): A vector with the train savings value along the epoch of the training
        """

        self.estimate_a(target=labels_train,  # Compute the values of a1 and a0
                        cr01=cr01,
                        cr10=cr10,
                        cr00=cr00,
                        cr11=cr11)
        q_neutral = ir_original / ir_objetive

        loss_train = numpy.zeros(num_epochs)
        loss_test = numpy.zeros(num_epochs)
        saving_train = numpy.zeros(num_epochs)
        saving_test = numpy.zeros(num_epochs)

        for epoch in tqdm.tqdm(range(num_epochs)):

            ind_Datos_suffle = numpy.random.choice(range(len(inputs_train)), len(inputs_train), replace=False)
            inputs_train = inputs_train[ind_Datos_suffle]
            labels_train = labels_train[ind_Datos_suffle]
            cr01 = cr01[ind_Datos_suffle]
            cr10 = cr10[ind_Datos_suffle]
            cr00 = cr00[ind_Datos_suffle]
            cr11 = cr11[ind_Datos_suffle]
            dataset_train = torch.utils.data.TensorDataset(inputs_train,
                                                           labels_train,
                                                           cr01,
                                                           cr10,
                                                           cr00,
                                                           cr11)
            trainloader_train = torch.utils.data.DataLoader(dataset_train,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=0)

            for i, data in enumerate(trainloader_train, 0):

                inputs_fold, labels_fold, cr01_fold, cr10_fold, cr00_fold, cr11_fold = data
                inputs_fold = inputs_fold.float()

                train_alphas_cost, train_alphas_RB = self.neutral_cost_weighted_alphas(target=labels_fold,
                                                                                       c01=cr01_fold.double(),
                                                                                       c10=cr10_fold.double(),
                                                                                       c00=cr00_fold.double(),
                                                                                       c11=cr11_fold.double(),
                                                                                       rebalance=q_neutral)
                optimizer.zero_grad()
                outputs = self(x_input=inputs_fold)

                loss = criterion(model_output=outputs.double(),
                                 target=labels_fold.view(len(outputs), 1).double(),
                                 alpha_cost=train_alphas_cost.double(),
                                 alpha_RB=train_alphas_RB.double())
                loss.backward()
                optimizer.step()

            if metrics is True:

                inputs_train_epoch = inputs_train.float()
                inputs_test = inputs_test.float()

                # ###### Train Metrics ###### #
                alphas_tr_cost, alphas_tr_RB = self.neutral_cost_weighted_alphas(target=labels_train,
                                                                                 c01=cr01.double(),
                                                                                 c10=cr10.double(),
                                                                                 c00=cr00.double(),
                                                                                 c11=cr11.double(),
                                                                                 rebalance=q_neutral)
                output_train = self(inputs_train_epoch)
                y_pred_train = self.predict_class(model_output=output_train, q_neutral=q_neutral)

                loss_train[epoch] = criterion(model_output=output_train.double(),
                                              target=labels_train.view(len(labels_train), 1).double(),
                                              alpha_cost=alphas_tr_cost.double(),
                                              alpha_RB=alphas_tr_RB.double()).detach().numpy()
                saving_train[epoch] = self.savings_metric(y_pred_class=y_pred_train.double(),
                                                          target_class=labels_train.double(),
                                                          c01=cr01.double(),
                                                          c10=cr10.double())
                # ########################### #

                # ###### Test Metrics ###### #
                alphas_ts_cost, alphas_ts_RB = self.neutral_cost_weighted_alphas(target=labels_test,
                                                                                 c01=cs01.double(),
                                                                                 c10=cs10.double(),
                                                                                 c00=cs00.double(),
                                                                                 c11=cs11.double(),
                                                                                 rebalance=q_neutral)
                output_test = self(inputs_test)
                y_pred_test = self.predict_class(model_output=output_test, q_neutral=q_neutral)

                loss_test[epoch] = criterion(model_output=output_test.double(),
                                             target=labels_test.view(len(labels_test), 1).double(),
                                             alpha_cost=alphas_ts_cost.double(),
                                             alpha_RB=alphas_ts_RB.double()).detach().numpy()
                saving_test[epoch] = self.savings_metric(y_pred_class=y_pred_test.double(),
                                                         target_class=labels_test.double(),
                                                         c01=cs01,
                                                         c10=cs10)
                # ########################### #

        if figure is True:

            fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(8, 4))

            ax1.plot(loss_train, 'b', label='Train')
            ax1.plot(loss_test, 'r', label='Test')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Training epochs')
            ax1.grid(color='grey', linestyle='-', linewidth=0.5)
            ax1.legend()

            ax2.plot(saving_train, 'b', label='Train')
            ax2.plot(saving_test, 'r', label='Test')
            ax2.set_title('Training Savings')
            ax2.set_xlabel('Training epochs')
            ax2.grid(color='grey', linestyle='-', linewidth=0.5)
            ax2.legend()
            fig.show()

        results = {'loss_train': loss_train,
                   'loss_test': loss_test,
                   'saving_train': saving_train,
                   'saving_test': saving_test
                   }

        return results

    def decision_cost(self,
                      y_pred_class: numpy.ndarray,
                      target_class: numpy.ndarray,
                      c01: numpy.ndarray,
                      c10: numpy.ndarray
                      ) -> float:
        """
        This function computes the costs of a given decision.

        Args:
            y_pred_class (numpy.ndarray): The predicted classes for each sample.
            target_class (numpy.ndarray): The true labels for each sample.
            c01 (numpy.ndarray): The cost C01 for each sample.
            c10 (numpy.ndarray): The cost C10 for each sample.

        Returns:
            cost (float): The cost produced by the decisions.
        """
        d01 = (y_pred_class == -1) & (target_class == 1)
        d10 = (y_pred_class == 1) & (target_class == -1)
        cost = (numpy.sum(c01[d01]) + numpy.sum(c10[d10]))

        return cost

    def savings_metric(self,
                       y_pred_class: torch.Tensor,
                       target_class: torch.Tensor,
                       c01: torch.Tensor,
                       c10: torch.Tensor
                       ) -> float:
        """
        This function computes the value of the metric Savings.

        Args:
            y_pred_class (torch.Tensor): The predicted classes for each sample.
            target_class (torch.Tensor): The true labels for each sample.
            c01 (torch.Tensor): The cost C01 for each sample.
            c10 (torch.Tensor): The cost C10 for each sample.

        Returns:
            savings (float): The value of the savings metric.
        """

        y_pred_class_aux = y_pred_class.detach().numpy().flatten()
        target_class_aux = target_class.detach().numpy().flatten()
        c01_aux = c01.detach().numpy().flatten()
        c10_aux = c10.detach().numpy().flatten()

        d01 = (target_class_aux == 1)
        cost_f0 = numpy.sum(c01_aux[d01])

        d10 = (target_class_aux == -1)
        cost_f1 = numpy.sum(c10_aux[d10])

        cost = self.decision_cost(y_pred_class=y_pred_class_aux,
                                  target_class=target_class_aux,
                                  c01=c01_aux,
                                  c10=c10_aux)
        cost_l = min(cost_f0, cost_f1)
        savings = (cost_l - cost)/cost_l

        return savings

    def metrics(self,
                x_input: torch.Tensor,
                target_class: torch.Tensor,
                c01: torch.Tensor,
                c10: torch.Tensor,
                rebalance: float
                ) -> typing.Dict:
        """
        This function computes all the metrics for a given dataset and labels.

        The function evaluates the model, computes the decided class for each sample and computes all the metrics.

        Args:
            x_input (torch.Tensor): The input data.
            target_class (torch.Tensor): The true labels for each sample.
            tensor_C01 (torch.Tensor): The cost C01 for each sample.
            tensor_C10 (torch.Tensor): The cost C01 for each sample.
            rebalance (float): The amount of rebalance applied during the training.

        Returns:
            coste_final (float): The cost of the decisions
            saving_final (float): The Savings of the decisions
            accuracy (float): The accuracy of the decisions
            o_pred_tensor (torch.Tensor): The output of the NN model
        """

        o_pred_tensor = self(x_input.float())
        target_class_aux = target_class.detach().numpy().flatten()
        c01_aux = c01.detach().numpy().flatten()
        c10_aux = c10.detach().numpy().flatten()

        y_pred_tensor = self.predict_class(model_output=o_pred_tensor, q_neutral=rebalance)
        y_pred = y_pred_tensor.detach().numpy().flatten()

        coste_final = self.decision_cost(y_pred_class=y_pred,
                                         target_class=target_class_aux,
                                         c01=c01_aux,
                                         c10=c10_aux)
        saving_final = self.savings_metric(y_pred_class=y_pred_tensor,
                                           target_class=target_class.double(),
                                           c01=c01,
                                           c10=c10)
        accuracy = sklearn.metrics.accuracy_score(y_true=target_class_aux.astype(int),
                                                  y_pred=y_pred.astype(int))

        results = {'coste_final': coste_final,
                   'saving_final': saving_final,
                   'accuracy': accuracy,
                   'o_pred_tensor': o_pred_tensor,
                   }

        return results
