# Summary

Experimental data used in:

Peterson, E. J. & Verstynen, T. D. A way around the exploration-exploitation dilemma. http://biorxiv.org/lookup/doi/10.1101/671362 (2019) doi:10.1101/671362.

# Major results - rerun list
`make exp455 exp457 exp497 exp526 exp498 exp527 exp562 exp473 exp502 exp496 exp525 exp472 exp501 exp475 exp504 exp474 exp503 exp528 exp539 exp387 exp408 exp385 exp409 exp386 exp410 exp412 exp485 exp514 exp388 exp411 exp484 exp513 exp487 exp516 exp486 exp515 exp531 exp542 exp547 exp548 exp549 exp550 exp551 exp552 exp553 exp573 exp574 exp554 exp555 exp556 exp557 exp575 exp576 exp558 exp559 exp560 exp561 exp383 exp403 exp381 exp404 exp382 exp405 exp407 exp481 exp510 exp384 exp406 exp480 exp509 exp483 exp512 exp482 exp511 exp530 exp541 exp391 exp449 exp389 exp450 exp390 exp451 exp453 exp489 exp518 exp392 exp452 exp488 exp517 exp491 exp520 exp490 exp519 exp532 exp543 exp460 exp462 exp458 exp463 exp459 exp466 exp493 exp522 exp461 exp465 exp492 exp521 exp495 exp524 exp494 exp523 exp533 exp544`

## BanditOneHigh4
Ours
- meta: exp457 (exp455_sorted)
Random
- epsilon: exp526 (exp497_sorted)
- decay: exp527 (exp498_sorted)
- random: exp562 (NA)
Reward
- extrinsic: exp502 (exp473_sorted)
Intrinsic
- info: exp525 (exp496_sorted)
- novelty: exp501 (exp472_sorted)
- entropy:  exp504 (exp475_sorted)
- EB:  exp503 (exp474_sorted)
- UCB:  exp539 (exp528_sorted)


## BanditHardAndSparse10
Ours
- meta: exp408 (exp387_sorted)
Random
- epsilon: exp409 (exp385_sorted)
- decay: exp410 (exp386_sorted)
- random: exp412 (NA)
Reward
- extrinsic: exp514 (exp485_sorted)
Intrinsic 
- info: exp411 (exp388_sorted)
- novelty: exp513 (exp484_sorted)
- entropy:  exp516 (exp487_sorted)
- EB: exp515 (exp486_sorted)
- UCB:  exp542 (exp531_sorted)


## BanditUniform121
Ours
- meta: exp548 (exp547_sorted)
Random
- epsilon: exp550 (exp549_sorted)
- decay: exp552 (exp551_sorted)
- random: exp553 (NA)
Reward
- extrinsic: exp574 (exp573_sorted)
Intrinsic
- info: exp555 (exp554_sorted)
- novelty: exp557 (exp556_sorted)
- entropy:  exp576 (exp575_sorted)
- EB: exp559 (exp558_sorted)
- UCB: exp561 (exp560_sorted)


## BanditUniform121 
Ours
- meta: exp403 (exp383_sorted)
Random
- epsilon: exp404 (exp381_sorted)
- decay: exp405 (exp382_sorted)
- random: exp407 (NA)
Reward
- extrinsic: exp510 (exp481_sorted)
Intrinsic
- info: exp406 (exp384_sorted)
- novelty: exp509 (exp480_sorted)
- entropy:  exp512 (exp483_sorted)
- EB:  exp511 (exp482_sorted)
- UCB:  exp541 (exp530_sorted)


## DeceptiveBanditOneHigh10
Ours
- meta: exp449 (exp391_sorted)
Random
- epsilon: exp450 (exp389_sorted)
- decay: exp451 (exp390_sorted)
- random: exp453 (NA)
Reward
- extrinsic: exp518 (exp489_sorted)
Intrinsic
- info: exp452 (exp392_sorted)
- novelty: exp517 (exp488_sorted)
- entropy: exp520 (exp491_sorted)
- EB: exp519 (exp490_sorted)
- UCB:  exp543 (exp532_sorted)


## DistractionBanditOneHigh10
Ours
- meta: exp462 (exp460_sorted)
Random
- epsilon: exp463 (exp458_sorted)
- decay: exp464 (exp459_sorted)
- random: exp466 (NA)
Reward
- extrinsic: exp522 (exp493_sorted)
Intrinsic
- info: exp465 (exp461_sorted)
- novelty: exp521 (exp492_sorted)
- entropy: exp524 (exp495_sorted)
- EB: exp523 (exp494_sorted)
- UCB:  exp544 (exp533_sorted)


# Alt. memory
## BanditOneHigh4 
Ours
- L1/Prob: exp689 (exp669_sorted)
- L2/Entropy: exp690 (exp670_sorted)
- L2/dRate: exp691 (exp671_sorted)
- UCB/Count exp692 (exp672_sorted)
- KL/Prob: exp693 (exp673_sorted)

## BanditUniform121 
Ours
- L1/Prob: exp694 (exp674_sorted)
- L2/Entropy: exp695 (exp675_sorted)
- L2/dRate: exp696 (exp676_sorted)
- UCB/Count exp697 (exp677_sorted)
- KL/Prob: exp698 (exp678_sorted)