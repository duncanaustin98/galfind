import click
# from dropbox import Dropbox

# from . import config


@click.command()
@click.option(
    "--survey", default="JOF", help="The survey to download data from."
)
@click.option(
    "--version",
    default="v11",
    help="The version of the survey to download data from.",
)
@click.option(
    "--instrument_name",
    default="ACS_WFC+NIRCam",
    help="The instrument used for the survey.",
)
@click.option(
    "--forced_phot_band",
    default="F277W+F356W+F444W",
    help="The band(s) used for forced photometry.",
)
@click.option(
    "--include", default="all", help="The data to include in the download."
)
@click.option(
    "--token", default=None, help="The dropbox token to use for the download."
)
def download_data(
    survey, version, instrument_name, forced_phot_band, include, token
):
    """Download data from the EPOCHS dropbox."""
    if token is not None:
        pass
        # dbx = Dropbox(token)


if __name__ == "__main__":
    download_data()
