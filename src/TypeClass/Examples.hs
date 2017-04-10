{-# LANGUAGE DeriveFunctor #-}

module TypeClass.Examples where

import           Data.Semigroup

unitFormsASemigroup :: ()
unitFormsASemigroup = () <> ()

sumFormsASemigroup :: Sum Int
sumFormsASemigroup = 2 <> 3

productFormsASemigroup :: Product Int
productFormsASemigroup = 2 <> 3

listsFormASemigroup :: [Int]
listsFormASemigroup = [1, 2] <> [3, 4]

stringsFormASemigroup :: String
stringsFormASemigroup = "Hello" <> " " <> "World"

semigroupsAreAssociative :: Bool
semigroupsAreAssociative = (2 <> (3 <> 4) :: Sum Int) == ((2 <> 3) <> 4 :: Sum Int) -- Associative law

unitHasMempty :: ()
unitHasMempty = mempty :: ()

sumHasMappend :: Sum Int
sumHasMappend = 2 `mappend` 3

productHasMappend :: Product Int
productHasMappend = 2 `mappend` 3

listHasMappend :: [Int]
listHasMappend = [1, 2] `mappend` [3, 4]

stringHasMappend :: String
stringHasMappend = "Hello" `mappend` " " `mappend` "World"

addingTwoNumbers :: Int
addingTwoNumbers = 1 + 2

addingTwoNumbersCurried :: Int
addingTwoNumbersCurried = (+) 1 2

incrementNumberWithFunctor :: Maybe Int
incrementNumberWithFunctor = (+1) <$> Just 2

incrementNumbersWithFunctor :: [Int]
incrementNumbersWithFunctor = (+1) <$> [1, 2, 3]

-- Deriving Functor.  Why does this work?
data Tree a = Leaf a | Node (Tree a) (Tree a)
  deriving (Show, Eq, Ord, Functor)

simpleTree :: Tree Int
simpleTree = Node (Node (Leaf 1) (Leaf 2)) (Leaf 3)

fmappedTree :: Tree Int
fmappedTree = (+1) <$> simpleTree

functionComposition1 :: Int
functionComposition1 = let f = (+2) . (*3) in f 1

functionComposition2 :: Int
functionComposition2 = let f = (+2) . (*3) . (+4) in f 1

functionAsAFunctor1 :: Int
functionAsAFunctor1 = let f = (+2) <$> (*3) in f 1

functionAsAFunctor2 :: Int
functionAsAFunctor2 = let f = (+2) <$> (*3) <$> (+4) in f 1

prependHiWithApplicative :: Maybe String
prependHiWithApplicative = ("Hi " ++) <$> Just "Everyone"

addTwoNumbersWithApplicative1 :: Maybe Int
addTwoNumbersWithApplicative1 = (+) <$> Just 2 <*> Just 3

addTwoNumbersWithApplicative2 :: Maybe Int
addTwoNumbersWithApplicative2 = (+) <$> Nothing <*> Just 3

addTwoNumbersWithApplicative3 :: Maybe Int
addTwoNumbersWithApplicative3 = (+) <$> Just 2 <*> Nothing

addTwoNumbersWithApplicative4 :: Maybe Int
addTwoNumbersWithApplicative4 = (+) <$> Nothing <*> Nothing

addNumbersFromListWithApplicative1 :: [Int]
addNumbersFromListWithApplicative1 = (+) <$> [1] <*> [10]

addNumbersFromListWithApplicative2 :: [Int]
addNumbersFromListWithApplicative2 = (+) <$> [] <*> [10]

addNumbersFromListWithApplicative3 :: [Int]
addNumbersFromListWithApplicative3 = (+) <$> [1] <*> []

addNumbersFromListWithApplicative4 :: [Int]
addNumbersFromListWithApplicative4 = (+) <$> [] <*> []

addNumbersFromListWithApplicative5 :: [Int]
addNumbersFromListWithApplicative5 = (+) <$> [1, 2, 3] <*> [10, 20, 30]

appendTwoStringsWithApplicative :: Maybe String
appendTwoStringsWithApplicative = (++) <$> Just "Hi " <*> Just "Everyone"

functionAsAnApplicative1 :: Int
functionAsAnApplicative1 = let f = (+) <$> (+3) <*> (+4) in f 1

functionAsAnApplicative3 :: Int
functionAsAnApplicative3 = let
  g = (+) <$> (+3)
  f = g <*> (+4)
  in f 1
