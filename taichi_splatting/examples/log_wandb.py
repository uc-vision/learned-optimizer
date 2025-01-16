def log_adam_behavior_to_wandb(gaussians, adam_optimizer, iter,
                               rendered_image):
    """
    Logs Adam optimizer behavior to wandb.
    """
    log_data = {"iter": iter}

    # Iterate over parameter groups to log weights and biases
    for idx, param_group in enumerate(adam_optimizer.param_groups):
        for param in param_group['params']:
            if param.grad is not None:
                if param.ndimension() > 1:  # Likely weights
                    log_data[f"param_group_{idx}/weights_mean"] = param.data.mean().item()
                    log_data[f"param_group_{idx}/weights_std"] = param.data.std().item()
                    log_data[f"param_group_{idx}/weights_grad_mean"] = param.grad.mean().item()
                    log_data[f"param_group_{idx}/weights_grad_std"] = param.grad.std().item()
                elif param.ndimension() == 1:  # Likely biases
                    log_data[f"param_group_{idx}/biases_mean"] = param.data.mean().item()
                    log_data[f"param_group_{idx}/biases_std"] = param.data.std().item()
                    log_data[f"param_group_{idx}/biases_grad_mean"] = param.grad.mean().item()
                    log_data[f"param_group_{idx}/biases_grad_std"] = param.grad.std().item()
    log_data[f"iter_{iter}/rendered_image"] = wandb.Image(
        rendered_image.cpu().numpy(), caption=f"Rendered Image at Iteration {iter}"
    )
    # Log to wandb
    wandb.log(log_data)