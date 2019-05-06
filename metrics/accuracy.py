import torch

def calculate_accuracy(outputs, targets):

    batch_size = targets.size(0)

    with torch.no_grad():
        _, predicted = outputs.topk(1, 1, True)
        assert predicted.shape[0] == len(targets)
        predicted = predicted.squeeze()  # from torch.size([batch_size, 1]) to torch.size([batch_size])
        n_correct_elems = (predicted == targets).float().sum().item()

    return n_correct_elems / batch_size


def calculate_accuracy_old(outputs, targets):

    """

    :param outputs: torch tensor batch_size x ?
    :param targets: torch tensor batch_size x ?
    :return: scalar
    """

    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)  # topk returns the k largest elements of the given input tensor along a given dimension.
    pred = pred.t()  # t() expects input to be a matrix (2-D tensor) and transposes dimensions 0 and 1.
    correct = pred.eq(targets.view(1, -1))
    # n_correct_elems = correct.float().sum().data[0]
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size