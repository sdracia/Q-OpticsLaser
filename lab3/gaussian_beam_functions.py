import os
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from PIL import Image
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# Extract red_channel
def extract_red(filename, noise = False, cz = False):
  img = Image.open(filename).convert("RGB")
  red_channel = np.array(img)[:, :, 0]
  red_channel = red_channel / 255.0
  
  if noise:
    red_channel= gaussian_filter(red_channel, sigma=1)
  
  # Center and zoom if enabled
  if cz:
    non_zero_indices = np.argwhere(red_channel > 0.03)
        
    if non_zero_indices.size > 0:
      # Bounding box of non-zero region
      min_y, min_x = non_zero_indices.min(axis=0)
      max_y, max_x = non_zero_indices.max(axis=0)
            
      # Center and calculate zoom size
      center_y = (min_y + max_y) // 2
      center_x = (min_x + max_x) // 2
      half_size_y = (max_y - min_y) // 2 + 10  # Add some padding
      half_size_x = (max_x - min_x) // 2 + 10  # Add some padding
            
      # Crop region with boundaries
      start_x = max(0, center_x - half_size_x)
      end_x = min(red_channel.shape[1], center_x + half_size_x)
      start_y = max(0, center_y - half_size_y)
      end_y = min(red_channel.shape[0], center_y + half_size_y)
            
      # Crop the image
      red_channel = red_channel[start_y:end_y, start_x:end_x]
    else:
      print("Warning: The red channel has no non-zero values. Returning original.")
        
  return red_channel

# Plot image
def visualize_image(image, title="Image", cmap="hot"):
  plt.imshow(image, cmap=cmap)
  plt.colorbar(label="Intensity")
  plt.title(title)
  plt.axis("off")
  plt.show()


def extract_images(directory_path):
  image_files = [f for f in os.listdir(directory_path) if f.endswith('.bmp')]
    
  images = []
  z_positions = []

  for _, image_file in enumerate(image_files, start=1):
    z_match = re.search(r'(\d+)', image_file)  # Looks for numbers in the file name
    if z_match:
      z_position = int(z_match.group(1))  # Convert the matched number to integer
      z_positions.append(z_position)
    else:
      raise ValueError(f"Could not extract z-position from file name: {image_file}")

    # Construct the full file path
    image_path = os.path.join(directory_path, image_file)
    image = extract_red(image_path, noise=True, cz=True)  
    images.append(image)

  z_positions = np.array(z_positions)

  # Sort images and z_positions together by z_positions
  sorted_indices = np.argsort(z_positions)
  z_positions = z_positions[sorted_indices]
  images = [images[i] for i in sorted_indices]

  return images, z_positions


# 2D Gaussian Function
def gaussian_2d(coords, I0, x0, y0, sigma_x, sigma_y):
  x, y = coords
  return I0 * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))).ravel()

# Fit Gaussian
def fit_gaussian(image):
  x = np.arange(image.shape[1])
  y = np.arange(image.shape[0])
  x, y = np.meshgrid(x, y)
  
  y0, x0 = np.unravel_index(np.argmax(image), image.shape)
  initial_guess = (image.max(), x0, y0, 10, 10)
  params, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=initial_guess)
  return params

# Visualize Fit
def visualize_fit(image, params):
  x = np.arange(image.shape[1])
  y = np.arange(image.shape[0])
  x, y = np.meshgrid(x, y)
  
  fit_image = gaussian_2d((x, y), *params).reshape(image.shape)
  plt.imshow(image, cmap="hot")
  plt.contour(fit_image, colors="cyan")
  plt.title("Gaussian Fit")
  plt.axis("off")
  plt.show()

# 1D Gaussian Function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def cross_sections(image, plot=False, verbose=False):
  center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
  horizontal = image[center_y, :]
  vertical = image[:, center_x]

  # Define x values
  x_horizontal = np.arange(len(horizontal))
  x_vertical = np.arange(len(vertical))

  # Fit Gaussian to horizontal and vertical profiles
  popt_horizontal, _ = curve_fit(gaussian, x_horizontal, horizontal, p0=[1, center_x, 10])
  popt_vertical, _ = curve_fit(gaussian, x_vertical, vertical, p0=[1, center_y, 10])
  
  scale=3.45e-4
  
  _, _, sigma_x = popt_horizontal
  W_x = 2 * sigma_x * scale
    
  _, _, sigma_y = popt_vertical
  W_y = 2 * sigma_y * scale
    
  if plot:
    # Generate Gaussian curves
    fitted_horizontal = gaussian(x_horizontal, *popt_horizontal)
    fitted_vertical = gaussian(x_vertical, *popt_vertical)

    plt.figure(figsize=(6, 12))

    # Horizontal cross-section
    plt.subplot(2, 1, 1)
    plt.plot(x_horizontal, horizontal, label="Horizontal data", color="orange")
    plt.plot(x_horizontal, fitted_horizontal, label="Gaussian fit", color="red", linestyle="--")
    plt.title("Horizontal cross-section")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity")
    plt.legend()

    # Vertical cross-section
    plt.subplot(2, 1, 2)
    plt.plot(x_vertical, vertical, label="Vertical data", color="cyan")
    plt.plot(x_vertical, fitted_vertical, label="Gaussian fit", color="red", linestyle="--")
    plt.title("Vertical cross-section")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity")
    plt.legend()

    plt.tight_layout()
    plt.show()
  
  if verbose:
    # Print Gaussian parameters
    print("Horizontal Fit Parameters: Amplitude = {:.2f}, Mean = {:.2f}, Std Dev = {:.2f}".format(*popt_horizontal))
    print("Vertical Fit Parameters: Amplitude = {:.2f}, Mean = {:.2f}, Std Dev = {:.2f}".format(*popt_vertical))

  return W_x, W_y

# Calculate Beam Parameters
def calculate_beam_parameters(params):
  I0, x0, y0, sigma_x, sigma_y = params
  
  return {
    "Peak Intensity": I0,
    "Center (x, y)": (x0, y0),
    "Beam Width (sigma_x)": sigma_x,
    "Beam Width (sigma_y)": sigma_y,
    "Ellipticity (sigma_x / sigma_y)": sigma_x / sigma_y
  }

# Fourier Analysis
def fourier_analysis(image):
  fft_image = np.fft.fftshift(np.fft.fft2(image))
  magnitude = np.abs(fft_image)
  plt.imshow(np.log1p(magnitude), cmap="viridis")
  plt.title("Fourier Transform (Log Magnitude)")
  plt.colorbar(label="Log Intensity")
  plt.axis("off")
  plt.show()
  return magnitude

# Total Power Distribution
def calculate_total_power(image):
  return np.sum(image)


def visualize_image_3D(image, title="3D Visualization", width=700, height=600):
  x = np.arange(image.shape[1])
  y = np.arange(image.shape[0])
  x, y = np.meshgrid(x, y)
  z = image

  fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="hot", colorbar=dict(title="Intensity"),)])
  
  fig.update_layout(
    title=title,
    scene=dict(xaxis_title="X-axis", yaxis_title="Y-axis", zaxis_title="Intensity",),
    margin=dict(l=0, r=0, t=40, b=40), width=width, height=height, 
    )
  
  fig.show()
  

def compute_beam_radius(images):
  beam_radii = []
    
  for image in images:
    W_x, W_y = cross_sections(image)
        
    W_avg = (W_x + W_y) / 2
    beam_radii.append(W_avg)
    
  return beam_radii


def gaussian_beam_model(z, W0, zR):
  return W0 * np.sqrt(1 + (z / zR) ** 2)

def gaussian_beam_model_with_lens(z, W0, zR, f):
  return W0 * np.sqrt(1 + ((z - f) / zR) ** 2)


def fit_beam_radii(images, z_positions, lens = False, plot=False):
  beam_radii = compute_beam_radius(images)
  
  if not lens: 
    popt, pcov = curve_fit(gaussian_beam_model, z_positions, beam_radii, p0=[min(beam_radii), 50])
    W0, zR = popt

    perr = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters
    W0_err, zR_err = perr

    # Print optimal parameters with errors
    print(f"Optimal parameters:")
    print(f"  W0: {W0:.4f} ± {W0_err:.4f} [cm]")
    print(f"  zR: {zR:.4f} ± {zR_err:.4f} [cm]")
    
  else:
    # Fit the data to the model
    popt, pcov = curve_fit(gaussian_beam_model_with_lens, z_positions, beam_radii, p0=[min(beam_radii), 50, np.mean(z_positions)])
    W0, zR, f = popt
    
    # Standard deviations of the parameters
    perr = np.sqrt(np.diag(pcov))
    W0_err, zR_err, f_err = perr
    
    # Print optimal parameters with errors
    print(f"Optimal parameters:")
    print(f"  W0: {W0:.4f} ± {W0_err:.4f} [cm]")
    print(f"  zR: {zR:.4f} ± {zR_err:.4f} [cm]")
    print(f"  f: {f:.4f} ± {f_err:.4f} [cm]")
  
  if plot:  
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(z_positions, beam_radii, color="red", label="Computed beam radii")
    z_fit = np.linspace(min(z_positions), max(z_positions), 500)
    
    if not lens:
      W_fit = gaussian_beam_model(z_fit, W0, zR)
      plt.plot(z_fit, W_fit, color="red", label=f"Fit: $W_0$={W0:.4f} ± {W0_err:.4f}, $z_R$={zR:.2f} ± {zR_err:.2f}")
    else:
      W_fit = gaussian_beam_model_with_lens(z_fit, W0, zR, f)
      plt.plot(z_fit, W_fit, color="blue", label=f"Fit: $f$={f:.4f} ± {f_err:.4f}")

    plt.xlabel("z (position) [cm]")
    plt.ylabel("W(z) (beam radius) [cm]")
    plt.title("Beam radius W(z) vs z")
    plt.legend()
    plt.grid()
    plt.show()
    
  return popt, pcov


def fresnel_number(z, a):
  l = 633e-7
  return a**2 / (l * z)

def fit_diffraction(fringes, z_positions, plot=False):
  popt, pcov = curve_fit(fresnel_number, z_positions, fringes, p0=[100])
  
  perr = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters
  
  # Print optimal parameters with errors
  print(f"Optimal parameters:")
  print(f"  a: {popt[0]:.4f} ± {perr[0]:.4f} [cm]")
    
  if plot:  
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(z_positions, fringes, color="orange", label="Computed Fresnel number")
    z_fit = np.linspace(min(z_positions), max(z_positions), 500)
    N_fit = fresnel_number(z_fit, popt[0])
    plt.plot(z_fit, N_fit, color="red", label=f"Fit: $a$={popt[0]:.4f} ± {perr[0]:.4f}")

    plt.xlabel("z (position) [cm]")
    plt.ylabel(f"$N_f$ (Fresnel number) [cm]")
    plt.title(f"Fresnel number $N_f$ vs z")
    plt.legend()
    plt.grid()
    plt.show()
    
  return popt, pcov