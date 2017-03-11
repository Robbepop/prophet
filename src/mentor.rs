
/// A Mentor is an object type that is able to train a Disciple
/// to become a fully trained Prophet.
/// The Mentor tries to accelerate the learning rate of the Disciple
/// by self-configuring the learning parameters during the training session.
struct Mentor {
	stats : ErrorStats,
	config: LearnConfig
}

impl Mentor {
	pub fn new(stats: ErrorStats, config: LearnConfig) -> Self {
		Mentor{
			stats: stats,
			config: config
		}
	}

	pub fn train(disciple: Disciple) -> Prophet {
		// TODO!
	}

	pub fn train_with_config(disciple: Disciple, config: LearnConfig) -> Prophet {
		// TODO!
	}
}
