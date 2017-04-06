# hw-examples

## Getting started
To get started you need to have Haskell compiler to be installed, as well as some IDE support.  

In this course we will use [stack](https://docs.haskellstack.org/en/stable/README/) - a Haskell build 
tool that also takes care of installing GHC (Haskell compiler) and package management. 

### Installing `stack`
- Create `~/.local/bin` and add it to your `PATH`.
- Install [stack](https://docs.haskellstack.org/en/stable/README/): 
  `curl -sSL https://get.haskellstack.org/ | sh` or `brew install haskell-stack`
- Watch if there were any warnings about `PATH`
- [Optional] Run `stack setup` and `stack ghci` to verify that you can run Haskell on your machine.

We will also use [Atom](https://atom.io/) as an IDE with some Haskell-related plugins. 

### IDE support
There are number of options to use as and IDE for Haskell. Most popular are `Emacs` (`Spacemacs`), `Atom` and `VSCode`. 
There is a `Haskforce` plugin for `Intellij Idea` too.

#### Setting up `Atom`
- Install [Atom](https://atom.io/) (click and follow the instructions)
- Install useful plugins: `apm install file-icons linter`
- Install Haskell dev tools: </br> `stack install ghc-mod hindent hlint stylish-haskell hasktags`.
- Verify installation: `ghc-mod --version`.
- Install `Atom` packages for Haskell: </br> `apm install ide-haskell language-haskell haskell-ghc-mod linter-hlint autocomplete-haskell ide-haskell-hasktags`.

#### Setting up `VSCode`
- Install [VSCode](https://code.visualstudio.com/)
- Open `VSCode` and make it register itself in `PATH`:
  - Click `CMD+SHIFT+P`
  - Type `shell`
  - Select `Install 'code' command in PATH`
- Install useful pluging:
  - in UI: `CMD+SHIFT+P` and select `Install Extensions`
  - Select `file-icons` and click `install`
- Install plugins for Haskell development:
  - `CMD+SHIFT+P`, `Install Extensions`
  - type `haskell` to filter a list
  - install `Haskell ghc-mod` and `haskell-linter`
  - look at https://marketplace.visualstudio.com/items?itemName=hoovercj.vscode-ghc-mod
  
  
  
