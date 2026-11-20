import torch
from torch.nn import functional as F

from src.training.train_switchable import answer_distillation_loss


def test_distillation_uses_next_answer_tokens_only_and_stops_teacher_gradients():
    student = torch.zeros(2, 5, 3, requires_grad=True)
    teacher = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3).requires_grad_()
    labels = torch.tensor([[-100, -100, 1, 2, -100], [-100, 1, -100, -100, -100]])
    # Three supervised next tokens, predicted at positions (0,1), (0,2), (1,0).
    expected_positions = torch.tensor([[False, True, True, False, False],
                                       [True, False, False, False, False]])
    temperature = 2.0
    expected = F.kl_div(
        F.log_softmax(student[expected_positions] / temperature, dim=-1),
        F.softmax(teacher[expected_positions] / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature ** 2

    loss = answer_distillation_loss(student, teacher, labels, temperature)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(student.grad.abs().sum(-1) > 0, expected_positions)
    assert teacher.grad is None


def test_distillation_on_fully_masked_batch_is_finite_zero():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.full((1, 3), -100)
    loss = answer_distillation_loss(logits, torch.zeros_like(logits), labels, 2.0)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
