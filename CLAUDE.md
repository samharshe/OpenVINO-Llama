we are working on refactoring the current codebase for a demo. the current goal is to drive down the cost of supporting a new model near zero.

the demo currently works with a single model; we want to make the interfaces sufficiently abstract that it easily works with any model. we have written tests to ensure that we don't break anything while refactoring. we will keep all tests green unless we need to drastically refactor some part of the codebase, at which point we will consider the situation individually and decide on the best approach; we would like to always write to some test or spec so that we don't waste time in confused debugging.

it is essential that we proceed incrementally, test code often, and work with exceptional precision as we make trustworthy baby steps toward the goal.

do not conflate planning and executing. you and I should discuss everything thoroughly. you should confirm with me that you have a perfect understanding of what code you are writing before you try to write it. I should never have to say, "no, that's not what I was thinking."

we will make this entire process completely painless thanks to our thoughtfulness, our insistence on clarifying all confusion up front, and our adherence to the plan.

our edits will be surgical. we will change as few lines as possible and prefer the most elegant solutions, modifying code only if doing so is unavoidable to make the progress we are trying to make.

at each step, we will consult and update @plan.md. we will never freestyle when writing code. rather, if we see that the next step of @plan.md does not make sense, we will stop, brainstorm, update @plan.md, and continue with the revised plan.