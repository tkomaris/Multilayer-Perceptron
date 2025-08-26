#!/bin/bash

function which_dl {
	if uname -s | grep -iqF Darwin; then
		echo -e "Miniconda3-latest-MacOSX-x86_64.sh"
	else
		echo -e "Miniconda3-latest-Linux-x86_64.sh"
	fi
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
function which_shell {
	if echo -e $SHELL | grep -iqF zsh; then
		echo -e "zsh"
	else
		echo -e "bash"
	fi
}
function when_conda_exist {
	echo -e "Checking 42-$USER environment: "
	if conda info --envs | grep -iqF 42-$USER; then
		echo -e "\e[33mDONE\e[0m\n"
		echo -e "Updating pip and installing requirements in 42-$USER...\n"
		conda run -n 42-$USER python -m pip install --upgrade pip
		if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
			conda run -n 42-$USER python -m pip install -r "$SCRIPT_DIR/requirements.txt"
		fi
	else
		echo -e "\e[31mKO\e[0m\n"
		echo -e "\e[33mCreating 42 environnment:\e[0m\n"
		conda update -n base -c defaults conda -y
		conda create --name 42-$USER python=3.11 jupyter numpy pandas pycodestyle matplotlib isort -y
		echo -e "Updating pip and installing requirements in 42-$USER...\n"
		conda run -n 42-$USER python -m pip install --upgrade pip
		if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
			conda run -n 42-$USER python -m pip install -r "$SCRIPT_DIR/requirements.txt"
		fi
	fi
}

function set_conda {
	MINICONDA_PATH="/goinfre/$USER/miniconda3"
	if [ -d "/goinfre" ]; then
		MINICONDA_PATH="/goinfre/$USER/miniconda3"
	else
		MINICONDA_PATH="/home/$USER/miniconda3/"
	fi
	CONDA=$MINICONDA_PATH"/bin/conda"
	PYTHON_PATH=$(which python)
	REQUIREMENTS="jupyter numpy pandas pycodestyle"
	SCRIPT=$(which_dl)
	MY_SHELL=$(which_shell)
	DL_LINK="https://repo.anaconda.com/miniconda/"$SCRIPT
	DL_LOCATION="/tmp/"
	echo -e "Checking conda: "
	TEST=$(conda -h 2>/dev/null)
	if [ $? == 0 ] ; then
		echo -e "\e[32mOK\e[0m\n"
		when_conda_exist
		echo -e "\e[33mLaunch the following command or restart your shell:\e[0m\n"
		if [ $MY_SHELL == "zsh" ]; then
			echo -e "\tsource ~/.zshrc\n"
		else
			echo -e "\tsource ~/.bash_profile\n"
		fi
		return
	fi
	echo -e "\e[31mKO\e[0m\n"
	if [ ! -f $DL_LOCATION$SCRIPT ]; then
		echo -e "\e[33mDonwloading installer:\e[0m\n"
		cd $DL_LOCATION
		curl -LO $DL_LINK
		cd -
	fi
	echo -e "\e[33mInstalling conda:\e[0m\n"
	sh $DL_LOCATION$SCRIPT -b -p $MINICONDA_PATH
	echo -e "\e[33mConda initial setup:\e[0m\n"
	$CONDA init $MY_SHELL
	$CONDA config --set auto_activate_base false
	echo -e "\e[33mCreating 42-$USER environnment:\e[0m\n"
	$CONDA update -n base -c defaults conda -y
	$CONDA create --name 42-$USER python=3.11 jupyter numpy pandas pycodestyle matplotlib isort -y
	echo -e "Updating pip and installing requirements in 42-$USER...\n"
	$CONDA run -n 42-$USER python -m pip install --upgrade pip
	if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
		$CONDA run -n 42-$USER python -m pip install -r "$SCRIPT_DIR/requirements.txt"
	fi
	echo -e "\e[33mLaunch the following command or restart your shell:\e[0m\n"
	if [ $MY_SHELL == "zsh" ]; then
		echo -e "\tsource ~/.zshrc\n"
	else
		echo -e "\tsource ~/.bash_profile\n"
	fi
}
set_conda
