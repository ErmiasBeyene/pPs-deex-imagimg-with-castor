"""Helper functions to plot cross-check distributions from simulation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns

CLASS_TEXT = {
    -1: "Unknown",
    1: "True",
    2: "Phantom scattered",
    3: "Detector scattered",
    4: "Random"
}


def get_class_text(class_number):
  """Convert a class number to a human-readable string.

  Args:
    class_number (str or int): class number, as an integer or as a string that represents the number (for instance, "3").

  Returns:
    A human-readable string representing the class.
  """
  try:
    if isinstance(class_number, str): class_number = int(class_number)
    return CLASS_TEXT[class_number]
  except (ValueError, KeyError):
    # ValueError: class_number does not have the correct type
    # KeyError: class_number is not a valid key
    # Default to the numerical representation of the class
    return str(class_number)


LABEL_FORMAT = "%s [%s]"


def format_unit(label, unit=None):
  """Format an axis label with a unit. This function is useful to ensure consistency between all plots.

  Args:
    label (str): the label of the axis, for instance "Energy".
    unit (str): the unit of the axis, for instance "keV".

  Returns:
    A formatted string representing the axis label.
  """
  if unit is None:
    return label
  return LABEL_FORMAT % (label, unit)


def plot_classes(df_classes, xlabel="Class", ylabel="Count"):
  """Plot a count histogram from a series of coincidence classes.

  Args:
    df_classes (Series): series of classes.
    xlabel (str): label of the x-axis.
    ylabel (str): label of the y-axis.
  """
  plt.figure()
  ax = sns.countplot(x=df_classes)  # plot the histogram

  ax.bar_label(
      container=ax.containers[0], fmt="%d"
  )  # add count number on top of the bars
  ax.set_xticklabels(
      [get_class_text(text.get_text()) for text in ax.get_xticklabels()]
  )  # convert x tick label from numbers to human-readable classes
  ax.ticklabel_format(
      style="plain", axis="y"
  )  # disable scientific notation on the y-axis

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)


def plot_counts(dataframes, param, bins=100, legend_labels=None, **ax_params):
  """
  Plot an histogram that compares counts for a given parameter across various dataframes.

  Args:
    dataframes (list[DataFrame]): array of dataframes to plot from.
    param (str): parameter (dataframe series) to plot.
    bins (int): number of histogram bins.
    legend_labels (list[str]): legend label in the same order as `dataframes`.
    ax_params (dict): additional keyword arguments that are passed to matplotlib.
  """
  if not isinstance(dataframes, list): dataframes = [dataframes]

  plt.figure()
  for df in dataframes:
    plt.hist(x=df[param], bins=bins)
    plt.gca().set(**ax_params)
  if legend_labels is not None:
    plt.legend(labels=legend_labels)


def plot_comparison(
    dataframes,
    paramx,
    paramy=None,
    titles=None,
    sns_params=None,
    **ax_params
):
  """Plot a one- or two-dimensional histogram for various dataframes.

  Args:
    dataframes (list[DataFrame]): array of dataframes to plot from.
    paramx (str): parameter (dataframe series) to plot along the x-axis.
    paramy (str): parameter (dataframe series) to plot along the y-axis. If None, then 1-D histogram is assumed.
    titles (list[str]): list of title to use for the various histograms, in the same order as `dataframes`.
    sns_params (dict): additional keyword arguments that are passed to seaborn.
    ax_params (dict): additional keyword arguments that are passed to matplotlib.
  """

  # if `xlabel` or `ylabel` are not set, default to `paramx` and `paramy`.
  if "xlabel" not in ax_params: ax_params["xlabel"] = paramx
  if "ylabel" not in ax_params:
    if paramy:
      ax_params["ylabel"] = paramy
    else:
      ax_params["ylabel"] = 'Counts'

  # if a single plot is wanted, these instructions ensure that the function accepts a dataframe and a title string as arguments (instead of lists of one element).
  if not isinstance(dataframes, list): dataframes = [dataframes]
  if not isinstance(titles, list): titles = [titles]

  if sns_params is None: sns_params = {}

  n_dataframes = len(dataframes)

  figsize = (6 * n_dataframes, 6)
  _, ax = plt.subplots(1, n_dataframes, figsize=figsize)

  for i in range(n_dataframes):

    current_ax = ax[i] if n_dataframes > 1 else ax

    df = dataframes[i]
    if paramy:
      sns.histplot(
          x=df[paramx], y=df[paramy], ax=current_ax, cbar=True, **sns_params
      )
    else:
      sns.histplot(x=df[paramx], ax=current_ax, cbar=True, **sns_params)

    current_ax.set(**ax_params)

    # use as many titles from the list as possible, then defaults to no title.
    try:
      current_ax.set_title(titles[i])
    except (TypeError, IndexError):
      # TypeError: titles was not set
      # IndexError: not enough titles were provided
      pass


def plot_all_comparisons(
    dataframes,
    parameters,
    parameter_labels=None,
    parameter_lims=None,
    titles=None,
    sns_params=None,
    **ax_params
):
  """Plot a comparison for all possible pairs in a set of parameters for a set of dataframes.

  Args:
    dataframes (list[DataFrame]): array of dataframes to plot from.
    parameters (list[str]): list of parameters (dataframe series) to plot.
    parameter_labels (list[str]): list of axis labels to use for parameters, in the same order as `parameters`.
    parameter_lims (list[str]): list of axis limits to use for parameters, in the same order as `parameters`.
    titles (list[str]): list of title to use for the various histograms, in the same order as `dataframes`.
    sns_params (dict): additional keyword arguments that are passed to seaborn.
    ax_params (dict): additional keyword arguments that are passed to matplotlib.
  """

  def read_parameter(i, parameter_array, default_value=None):
    # Try to read parameter label
    # If not provided by the user, default to parameter key
    # TypeError: parameter_labels is None
    # IndexError: not enough labels provided
    try:
      return parameter_array[i]
    except (TypeError, IndexError):
      return default_value

  parameter_number = len(parameters)
  for px in range(parameter_number):
    for py in range(parameter_number):

      # Plot each comparison once (<)
      # and do not plot a parameter against itself (=)
      if px <= py:
        continue

      px_name = parameters[px]
      py_name = parameters[py]

      px_label = read_parameter(px, parameter_labels, px_name)
      py_label = read_parameter(py, parameter_labels, py_name)

      px_lim = read_parameter(px, parameter_lims)
      py_lim = read_parameter(py, parameter_lims)

      plot_comparison(
          dataframes,
          px_name,
          py_name,
          titles=titles,
          xlabel=px_label,
          ylabel=py_label,
          xlim=px_lim,
          ylim=py_lim,
          sns_params=sns_params,
          **ax_params
      )
