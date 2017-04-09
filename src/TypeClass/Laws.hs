{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE IncoherentInstances  #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

module TypeClass.Laws where

import           Data.Semigroup
import           Prelude        hiding (id)

{-# ANN module ("HLint: ignore"        :: String) #-}

id :: a -> a
id x = x

infixl 0 ===

class Equivalence a where
    (===) :: a -> a -> Bool

instance Eq a => Equivalence a where
    (===) = (==)

semigroupLaw    :: (Semigroup a, Eq a) => a -> a -> a -> Bool
monoidLaw0      :: (Monoid a, Eq a) => a -> Bool
monoidLaw1      :: (Monoid a, Eq a) => a -> Bool
monoidLaw2      :: (Monoid a, Eq a) => a -> a -> a -> Bool
functorLaw0     :: (Functor f, Eq (f b)) => f b -> Bool
functorLaw1     :: (Functor f, Eq (f b)) => (a -> b1) -> (b1 -> b) -> f a -> Bool
applicativeLaw0 :: (Applicative f, Eq (f a)) => f a -> Bool
applicativeLaw1 :: (Applicative f, Eq (f b)) => f a -> (a -> b) -> a -> Bool

semigroupLaw x y z = x <> (y <> z) === (x <> y) <> z

monoidLaw0 x        =           mempty `mappend` x === x
monoidLaw1 x        =           x `mappend` mempty === x
monoidLaw2 x y z    =  x `mappend` (y `mappend` z) === (x `mappend` y) `mappend` z

functorLaw0 x       =      fmap id x === id x                               -- Identity
functorLaw1 f g x   = fmap (g . f) x === (fmap g . fmap f) x                -- Fusion

applicativeLaw0 v       =              pure id <*> v === v                  -- Identity
applicativeLaw1 _ f x   = (pure f :: f (a -> b)) <*> pure x === pure (f x)         -- Homomorphism
applicativeLaw2 u y     =               u <*> pure y === pure ($ y) <*> u   -- Interchange
applicativeLaw3 u v w   = pure (.) <*> u <*> v <*> w === u <*> (v <*> w)    -- Composition

moo :: Applicative f => (a -> b) -> a -> f b
moo f x = pure f <*> pure x
