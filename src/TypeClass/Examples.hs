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
