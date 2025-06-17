from openpi.training import config
from openpi.policies import policy_config
from openpi.policies import droid_policy
from openpi.shared import download

def test_installation():
    print("Testing openpi installation...")
    
    # Get config and download checkpoint
    print("Loading model config...")
    config = config.get_config("pi0_fast_droid")
    print("Downloading checkpoint...")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")
    
    # Create policy
    print("Creating policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    
    # Run inference
    print("Running inference test...")
    example = droid_policy.make_droid_example()
    result = policy.infer(example)
    
    # Clean up
    del policy
    
    print("\nInstallation test successful!")
    print(f"Generated actions shape: {result['actions'].shape}")

if __name__ == "__main__":
    test_installation() 