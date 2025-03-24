# Guess for SUHF

First you need to determine the system is RHF stable or unstable, by doing RHF/UHF or by experience.

* RHF unstable means we need to start from converged UHF orbitals (not UNO)
  + Usual cases of doublet, triplet, etc., use `guess.gen(xyz, bas, charge, spin)`
  + Spin-polarized singlet, use `guess.mix(xyz, bas, charge, conv='tight')`
  + Hard cases of spin-polarized singlet, use `guess.from_frag(xyz, bas, frags, chgs, spins, cycle=50)`
  + Crazy cases. You can try `guess.from_fch_simp` to import a set of orbitals from Gaussian.
* RHF stable cases. I have no good idea for a general strategy. 
  + No static correlation cases. Just use RHF orbitals, and SUHF NO will be the same as RHF MO. 
  + Some cases like N2 at 1.1A. We need unconverged UHF orbitals here. Use `guess.mix(xyz, bas, charge)` or `guess.from_frag(xyz, bas, frags, chgs, spins)`. Note that default setting of `mix` and `from_frag` is very loose converging (`conv='loose'`), it's designed for this kind of cases.
  + For H2 at 0.75A. `conv='loose'` still cannot prevent converging to RHF wavefunction. We need `guess.mix(xyz, bas, cycle=0)`.

More tips
* Every functions mentioned above return a UHF object.
  + Even `gen(xyz, bas, 0, 0)` returns a UHF object. 
  + Use PySCF utilities `mf.to_rhf()` or `mf.to_uhf()` when needed.
* When using `guess.gen` or `guess.from_frag` for UHF cases, check UHF internal stability by `mf = guess.check_stab(mf)`. That's something like 'stable=opt'. 
  + `guess.mix(conv='tight') will do `check_stab` automatically.
  + When using default `mix` or `from_frag`, there's no need to perform `check_stab`, because the orbitals are not converged at all.
* use `mf.level_shift = 0.2` to activate level shift. 0.2-0.5 is recommended. Dynamic shift not implemented yet.
* DIIS is used by default.
