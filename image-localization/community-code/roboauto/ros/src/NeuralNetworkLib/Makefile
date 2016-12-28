include ./Makefile.const

.PHONY: documentation clean

documentation:
	doxygen

clean:
	@make -C src clean
	@make -C tests clean
	#@rm -f ./*.so ./*.a  ./*.nm
	@rm -f ./lib/*.so ./lib/*.a  ./lib/*.nm
	
	@echo "Cleaned....."
