import torch
import types
from typing import List
import sklearn.metrics
from .data import (
    GermEval17, GermEval18, GermEval19, GermEval21, GermEval21vmwe,
    MillionSentiment, MillionBinary, SBCHisSwiss, SBCHsenti, ArchiMob, LSDC)
from collections import Counter


class ClassiferModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,  # nclasses
                 bias: bool = True,
                 *args, **kwargs):
        super(ClassiferModel, self).__init__(*args, **kwargs)
        self.bias = bias
        self.final = torch.nn.Linear(
            input_size, output_size, bias=bias)
        self.soft = torch.nn.Softmax(dim=1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.manual_seed(42)
        torch.nn.init.xavier_normal_(self.final.weight, gain=1.0)
        if self.bias:
            torch.nn.init.zeros_(self.final.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.final(inputs)
        return self.soft(x)


def build_model(**kwargs):
    return ClassiferModel(
        input_size=kwargs['n_features'],
        output_size=kwargs['n_classes'])


def evaluate(downstream_tasks: List[str],
             preprocesser: types.FunctionType,
             modelbuilder: types.FunctionType = None,
             bias: bool = True,
             datafolder: str = "./datasets",
             batch_size: int = 64,
             num_epochs: int = 20,
             balanced: bool = False,
             early_stopping: bool = False,
             split_ratio: float = 0.2,
             patience: int = 5):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start
    results = []
    for downstream_task in downstream_tasks:
        # load datasets
        if downstream_task == "ABSD-1":
            ds_train = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Relevance", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Relevance", test=True)
        elif downstream_task == "ABSD-2":
            ds_train = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Sentiment", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Sentiment", test=True)
        elif downstream_task == "ABSD-3":
            ds_train = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Category", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval17(
                preprocesser, datafolder=datafolder,
                task="Category", test=True)

        elif downstream_task == "OL18-A":
            ds_train = GermEval18(
                preprocesser, datafolder=datafolder,
                task="A", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval18(
                preprocesser, datafolder=datafolder,
                task="A", test=True)
        elif downstream_task == "OL18-B":
            ds_train = GermEval18(
                preprocesser, datafolder=datafolder,
                task="B", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval18(
                preprocesser, datafolder=datafolder,
                task="B", test=True)

        elif downstream_task == "OL19-A":
            ds_train = GermEval19(
                preprocesser, datafolder=datafolder,
                task="A", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval19(
                preprocesser, datafolder=datafolder,
                task="A", test=True)
        elif downstream_task == "OL19-B":
            ds_train = GermEval19(
                preprocesser, datafolder=datafolder,
                task="B", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval19(
                preprocesser, datafolder=datafolder,
                task="B", test=True)
        elif downstream_task == "OL19-C":
            ds_train = GermEval19(
                preprocesser, datafolder=datafolder,
                task="C", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval19(
                preprocesser, datafolder=datafolder,
                task="C", test=True)

        elif downstream_task == "TOXIC":
            ds_train = GermEval21(
                preprocesser, datafolder=datafolder,
                task="TOXIC", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval21(
                preprocesser, datafolder=datafolder,
                task="TOXIC", test=True)
        elif downstream_task == "ENGAGE":
            ds_train = GermEval21(
                preprocesser, datafolder=datafolder,
                task="ENGAGE", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval21(
                preprocesser, datafolder=datafolder,
                task="ENGAGE", test=True)
        elif downstream_task == "FCLAIM":
            ds_train = GermEval21(
                preprocesser, datafolder=datafolder,
                task="FCLAIM", test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval21(
                preprocesser, datafolder=datafolder,
                task="FCLAIM", test=True)

        elif downstream_task == "VMWE":
            ds_train = GermEval21vmwe(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = GermEval21vmwe(
                preprocesser, datafolder=datafolder,
                test=True)

        elif downstream_task == "MIO-S":
            ds_train = MillionSentiment(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = MillionSentiment(
                preprocesser, datafolder=datafolder,
                test=True)
        elif downstream_task in ['MIO-O', 'MIO-I', 'MIO-D', 'MIO-F', 'MIO-P',
                                 'MIO-A']:
            ds_train = MillionBinary(
                preprocesser, datafolder=datafolder,
                test=False, task=downstream_task, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = MillionBinary(
                preprocesser, datafolder=datafolder,
                test=True, task=downstream_task)

        elif downstream_task == "SBCH-L":
            ds_train = SBCHisSwiss(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = SBCHisSwiss(
                preprocesser, datafolder=datafolder,
                test=True)
        elif downstream_task == "SBCH-S":
            ds_train = SBCHsenti(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = SBCHsenti(
                preprocesser, datafolder=datafolder,
                test=True)

        elif downstream_task == "ARCHI":
            ds_train = ArchiMob(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = ArchiMob(
                preprocesser, datafolder=datafolder,
                test=True)

        elif downstream_task == "LSDC":
            ds_train = LSDC(
                preprocesser, datafolder=datafolder,
                test=False, split_ratio=split_ratio,
                early_stopping=early_stopping)
            ds_test = LSDC(
                preprocesser, datafolder=datafolder,
                test=True)
        else:
            raise Exception(
                f"Downstream task '{downstream_task}' not available.")

        # data loader
        X_valid, y_valid = ds_train.get_validation_set()
        n_classes = ds_train.num_classes()
        n_features = ds_train.num_features()
        dgen = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size, shuffle=False)

        # init new model
        if not isinstance(modelbuilder, types.FunctionType):
            modelbuilder = build_model

        model = modelbuilder(
            n_features=n_features,
            n_classes=n_classes,
            bias=bias
        ).to(device)

        # loss function
        if balanced:
            class_weights = ds_train.get_class_weights()
        else:
            class_weights = torch.ones(n_classes) / n_classes

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='mean'
        ).to(device)

        # early stopping
        if y_valid is not None:
            with torch.no_grad():
                X_valid, y_valid = X_valid.to(device), y_valid.to(device)
                valid_loss = loss_fn(model(X_valid), y_valid).item()
                wait = 0

        # optimization settings
        optimizer = torch.optim.Adam(
            model.parameters(), lr=3e-4, betas=(.9, .999),
            eps=1e-7, amsgrad=True)

        for epoch in range(num_epochs):
            # train
            # epoch_loss = 0.
            for X_train, y_train in dgen:
                X_train, y_train = X_train.to(device), y_train.to(device)
                # train it
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item()
            # early stopping
            if y_valid is not None:
                with torch.no_grad():
                    tmp = loss_fn(model(X_valid), y_valid).item()
                    if tmp >= valid_loss:
                        wait += 1
                    else:
                        valid_loss = tmp
                        wait = 0
                    if wait > patience:
                        break

        # load datasets
        dgen_test = torch.utils.data.DataLoader(
            ds_test, batch_size=len(ds_test), shuffle=False)
        for X_test, y_test in dgen_test:
            X_test, y_test = X_test.to(device), y_test.to(device)
            break

        dgen_train = torch.utils.data.DataLoader(
            ds_train, batch_size=len(ds_train), shuffle=False)
        for X_train, y_train in dgen_train:
            X_train, y_train = X_train.to(device), y_train.to(device)
            break

        # compute metrics
        y_pred = torch.argmax(model(X_test), dim=1)
        res_test = {
            "num": len(y_pred),
            "acc": sklearn.metrics.accuracy_score(y_test, y_pred),
            "acc-balanced": sklearn.metrics.balanced_accuracy_score(
                y_test, y_pred),
            "f1": sklearn.metrics.f1_score(y_test, y_pred, average='micro'),
            "f1-balanced": sklearn.metrics.f1_score(
                y_test, y_pred, average='macro'),
            "distr-pred": dict(Counter(y_pred.detach().numpy())),
            "distr-test": dict(Counter(y_test.detach().numpy())),
        }
        y_pred = torch.argmax(model(X_train), dim=1)
        res_train = {
            "num": len(y_pred),
            "acc": sklearn.metrics.accuracy_score(y_train, y_pred),
            "acc-balanced": sklearn.metrics.balanced_accuracy_score(
                y_train, y_pred),
            "f1": sklearn.metrics.f1_score(y_train, y_pred, average='micro'),
            "f1-balanced": sklearn.metrics.f1_score(
                y_train, y_pred, average='macro'),
            "distr-pred": dict(Counter(y_pred.detach().numpy())),
            "distr-train": dict(Counter(y_train.detach().numpy())),
        }
        # save results
        results.append({
            "task": downstream_task,
            "epochs": epoch + 1,
            "test": res_test,
            "train": res_train
        })

    # done
    return results
