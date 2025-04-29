
# AEdiffusion_surrogate_DA_inversion
This is the code for the paper titled *Integration of DDPM and ILUES for Simultaneous Identification of Contaminant Source Parameters and Non-Gaussian Channelized Hydraulic Conductivity Field*.

### high_fidelity:
The `high_fidelity` folder contains parallel computing files for high-fidelity groundwater models, including MODFLOW and MT3DMS. These models will be called through the `.bat` file in the MATLAB `model_H` script and executed via CPU parallel computing to simulate the results.

### main:
The `TI` YAML files in the `Config` folder contain all hyperparameters used by AEdiffusion, including:
- **DDPM config** used for DDPM training
- **VAE config** used for VAE training
- All parameters for testing

The `latent.py` file contains all operations related to latent parameters.

The `TI_dataset.py` file describes how to import the geological implementation dataset that has been cut using TI.

The `sample_cond_surrogate.py` file contains the instructions for calling the AEdiffusion-ILUES-ARNW inversion framework. It is important to note that this code includes calls to the pre-trained AEdiffusion, ARNW, and the data assimilation algorithm ILUES, interspersed with data reading and saving operations.

The `models` folder contains VAE and DDPM network structures, which are identical to traditional standard architectures. The Unet here is based on OpenAI's repository code: https://github.com/openai/improved-diffusion

The `train_ae` and `train_ddpm` files are used for training the VAE and DDPM models, respectively.

### script:
- `AR_Net_training`: The code for training the ARNW model.
- `forward_model`: The code for running the groundwater forward simulation in MATLAB.
- `ilues`: The data assimilation algorithm. [Link to algorithm paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017WR020906).
- `model_H`: The parallel execution code for groundwater forward simulations.
- `obscoor`: Observation well coordinates.
- `RouletteWheelSelection`: ILUES's update strategy.
- `run_this`: Executes the forward model in MATLAB.
- `update_samples`: Part of the function in the ILUES algorithm.

### utility:
- `Cal_Log_Lik`: Calculates the log likelihood.
- `genex`: Generates random variables in the prior space.
- `getint`: Retrieves the integer-based coordinates.
- `local_update`: Part of the function in the ILUES algorithm.
- `modifyssm`: Modifies contaminant source parameters.
- `readMT3D`: Reads contaminant concentration information.
- `updatapara`: Part of the function in the ILUES algorithm.

