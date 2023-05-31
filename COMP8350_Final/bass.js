Clock.bpm = 160
r = Reverb().bus()
r_drum = Reverb().bus()
drum = Drums().connect(r_drum,2)
s2 = Synth('pwm.squeak').connect(r        ,1)
mod2 = gen( cycle( beats(4)*8 )* 4 )
mod2.connect( s2.frequency )

s3 = Synth('pwm.squeak').connect(r        ,1)
mod3 = gen( cycle( beats(4)*8 )* 4 )
mod3.connect( s3.frequency )

verb = Reverb( 'space' ).bus()
glitch = Freesound[5]({ query:'pumping', max:.5 })
  .connect( verb, .05 )
  .spread(1) // pan voices full stereo
bass2 = FM('deepbass', { decay:3 })



// learning more about genish will be helpful if
// you want to create these types of custom modulations.


drum.tidal('kd oh')

Pluck().note.seq( Rndi(0,4), 1/8 )

bass2.note.seq( [-4,-2,-1,0], 1 )

glitch.pickplay.seq(
  Rndi(0,14),
  Euclid(9,16)
)

s2.note.seq( sine(3,2,3)      , [1/8,1/4,1/8,1/4,1/4,1/3,1/3,1/3])
s3.note.seq( sine(2,1,1) , [3/8,3/8,1/4,1/3,1/3,1/3])

s2.stop()
s3.stop()
rev = Reverb('space').bus()
harm = Synth[5]('',{filterMode:1,gain:.5,decay:.5,attack:.001,Q:.7,useADSR:true})
chr = Chorus('lush').bus()
flg=Flanger('moderate').bus()
dis=Distortion('earshed').bus()
del=Delay('1/3').bus().connect(rev,0.1)

harm.connect(rev, .2)
harm.connect(chr, .0)
harm.connect(flg, .0)
harm.connect(dis, .0)
harm.connect(del, .3)
harm.attack = 0.005
harm.decay = .3
harm.sustain = .1
harm.gain = 1
harm.spread(0.25)
harm.pulsewidth = 0.5
//chord progressions
//intro + outro
//Theory.degree.seq(['-ii','-IV','-I','-
harm.chord.seq([[-6],[-4,0,2,5],[-4],[2]],1)
//middle
//Theory.degree.seq( ['IV','V','iii','vi
harm.chord.seq([[-4],[-2,0],[-3],[-1,1]],1)
//fades and stopping
harm.gain.fade(null,1,4)