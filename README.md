# hw-examples

## Getting started
To get started you need to have Haskell compiler to be installed, as well as some IDE support.  

In this course we will use [stack](https://docs.haskellstack.org/en/stable/README/) - a Haskell build 
tool that also takes care of installing GHC (Haskell compiler) and package management. 

We will also use [Atom](https://atom.io/) as an IDE with some Haskell-related plugins. 


#### Installing `stack`
- Create `~/.local/bin` and add it to your `PATH`.
- Install [stack](https://docs.haskellstack.org/en/stable/README/): 
  `curl -sSL https://get.haskellstack.org/ | sh` or `brew install haskell-stack`
- Watch if there were any warnings about `PATH`
- [Optional] Run `stack setup` and `stack ghci` to verify that you can run Haskell on your machine.

#### Installing `Atom`
- Install [Atom](https://atom.io/) (click and follow the instructions)
- Install useful plugins: `apm install file-icons linter`

#### Configuring `Atom` for `Haskell`
- Install Haskell dev tools: </br> `stack install ghc-mod hindent hlint stylish-haskell hasktags`.
- Verify installation: `ghc-mod --version`.
- Install `Atom` packages for Haskell: </br> `apm install ide-haskell language-haskell haskell-ghc-mod linter-hlint autocomplete-haskell ide-haskell-hasktags`.
